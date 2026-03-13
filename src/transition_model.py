"""Count-based transition model with integrated intrinsic reward computation.

The model maintains a Dirichlet count matrix over observed transitions
``(s, a, s')`` and derives the transition probability distribution
``p(s'|s, a)`` as the Dirichlet mean (or a sample from the posterior).

Intrinsic reward signals are computed incrementally after each update and
stored in normalized reward arrays so that the planning agent can query
``model.R`` directly.  Supported reward types:

- ``"novelty"``    – count-based novelty (−log visit-frequency)
- ``"empowerment"`` – 1-step channel capacity (Blahut–Arimoto)
- ``"info_gain"``  – predicted information gain (Little & Sommer, 2011)

Multiple rewards are combined by either averaging (``"mean"``) or multiplying
(``"product"``) their scaled [0, 1] components.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict

from intrinsic.info_gain import (
    calculate_predicted_information_gain_for_state_action_pair,
)
from intrinsic.empowerment import compute_empowerment_for_state
from intrinsic.novelty import compute_novelty_for_state, compute_novelty_for_all_states
from reward_scaler import GlobalRewardScaler


class CountBasedTransitionModel:
    """Transition model based on counting observed transitions.

    The counts array is initialised with a small *prior_count* (Laplace
    smoothing) so that unobserved transitions receive a non-zero probability.
    When a new transition is observed the count is incremented by
    *update_strength* (defaulting to ``100 * |S|``), which controls how
    quickly the model concentrates probability mass on observed outcomes.

    Args:
        num_actions: Number of actions in the environment.
        states: List of all possible states in the environment.
        prior_count: Initial count for all transitions (Laplace prior).
        update_strength: Count increment per observed transition.
            Defaults to ``100 * len(states)``.
        reward_types: Intrinsic reward signals to maintain.  Any subset of
            ``["novelty", "info_gain", "empowerment"]``.
        reward_configs: Per-reward keyword arguments forwarded to the
            corresponding reward computation function, keyed by reward name.
        combination_method: How to combine multiple scaled reward components
            into a single ``R`` matrix.  ``"mean"`` or ``"product"``.
        is_true_model: When ``True`` the model is treated as final and
            ``update()`` becomes a no-op for the count/reward arrays.
    """

    def __init__(
        self,
        num_actions: int,
        states: list,
        prior_count: float = 1,
        update_strength: float | None = None,
        reward_types: list[str] = [],
        reward_configs: dict[str, dict] | None = None,
        combination_method: str = "mean",
        is_true_model: bool = False,
    ) -> None:
        # All states
        self.observed_states = states
        self.state_to_idx: dict = {s: i for i, s in enumerate(states)}

        self.prior_count = prior_count
        self.update_strength = 100 * len(states) if update_strength is None else update_strength
        self.num_actions = num_actions

        # Counts[s, a, s'] stores the (smoothed) number of times transition
        # (s, a) → s' has been observed.
        self.counts = np.ones((len(states), num_actions, len(states))) * prior_count

        # Boolean mask: True for transitions that end an episode (lava/goal).
        self.terminal_transitions = np.zeros((len(states), num_actions, len(states)), dtype=bool)

        # Aggregate (intrinsic) reward used by the planning agent.
        self.R = np.zeros((len(states), num_actions, len(states)))

        self.reward_types = reward_types
        self.combination_method = combination_method
        self.reward_configs = reward_configs if reward_configs is not None else {}

        # Compute initial reward estimates under the prior model
        self.state_visit_counts = np.zeros(len(states), dtype=int)
        self.visited_states = set()
        self.discovery_order = []              # first time each state was discovered
        self.state_visit_history = []          # every visited state in temporal order
        self.discovery_timestep = {}           # state -> first timestep it was seen
        self.total_state_visits = 0

        self._initialize_intrinsic_rewards()
        self.is_true_model = is_true_model

    def get_discovered_states(self):
        return list(self.discovery_order)

    def get_num_discovered_states(self) -> int:
        return len(self.visited_states)

    def has_visited_state(self, state) -> bool:
        return state in self.visited_states

    def get_state_visit_count(self, state) -> int:
        return int(self.state_visit_counts[self.state_to_idx[state]])

    def get_discovery_timestep(self, state):
        return self.discovery_timestep.get(state, None)

    def _record_state_visit(self, state) -> None:
        s = self.state_to_idx[state]

        self.state_visit_counts[s] += 1
        self.state_visit_history.append(state)

        if state not in self.visited_states:
            self.visited_states.add(state)
            self.discovery_order.append(state)
            self.discovery_timestep[state] = self.total_state_visits

        self.total_state_visits += 1
        
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        state,
        action: int,
        next_state,
        extrinsic_reward: float = 0.0,
        terminated: bool = False,
    ) -> None:
        """Record an observed transition and refresh intrinsic reward estimates.

        Args:
            state: Current state (hashable, must be in ``observed_states``).
            action: Action taken.
            next_state: Resulting state.
            extrinsic_reward: Environment reward (not yet used).
            terminated: Whether the episode ended after this transition.
        """
        s = self.state_to_idx[state]
        a = action
        s_prime = self.state_to_idx[next_state]

        # --- NEW: record visit to resulting state ---
        self._record_state_visit(next_state)

        if not self.is_true_model:
            self.counts[s, a, s_prime] += self.update_strength
            if terminated:
                self.terminal_transitions[s, a, s_prime] = True

            # Recompute reward signals that depend on the updated counts
            self._update_intrinsic_rewards(state, action, next_state)

        # TODO: Implement extrinsic reward handling

    def get_full_transition_matrix(
        self, dirichlet: bool = False, random_state: int = 42
    ) -> tuple[np.ndarray, dict]:
        """Return the transition probability matrix ``p(s'|s, a)``.

        Args:
            dirichlet: If ``True``, sample a single draw from the Dirichlet
                posterior instead of returning the mean.
            random_state: Seed for the Dirichlet sampler (only used when
                ``dirichlet=True``).

        Returns:
            matrix: Array of shape ``(|S|, |A|, |S|)`` with transition
                probabilities summing to 1 along the last axis.
            state_to_idx: Mapping from states to row/column indices.
        """
        if not dirichlet:
            # Mean of the Dirichlet posterior (normalised counts)
            matrix = self.counts / np.sum(self.counts, axis=2, keepdims=True)

            # Round to reduce numerical noise from extremely small values
            matrix = np.round(matrix, decimals=2)

            # Renormalise after rounding to ensure rows sum to exactly 1.
            # Guard against degenerate all-zero rows (set to uniform).
            matrix[np.sum(matrix, axis=2) == 0] = 1
            matrix = matrix / np.sum(matrix, axis=2, keepdims=True)
        else:
            # Sample from the Dirichlet via the Gamma trick (vectorised)
            rng = np.random.default_rng(random_state)
            G = rng.gamma(shape=self.counts, scale=1.0)
            matrix = G / G.sum(axis=2, keepdims=True)

        return matrix, self.state_to_idx

    def get_true_counts(self) -> np.ndarray:
        """Return the net observed transition counts, stripped of the prior.

        Returns:
            Array of shape ``(|S|, |A|, |S|)`` where entry ``[s, a, s']`` is
            ``(counts[s,a,s'] − prior_count) / update_strength``, i.e. the
            number of times transition ``(s, a) → s'`` was observed.
        """
        raw_counts = (self.counts - self.prior_count) / self.update_strength
        return raw_counts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_intrinsic_rewards(self, state, action: int, next_state) -> None:
        """Incrementally refresh intrinsic reward components after an update.

        Only the rewards that depend on the most recently changed transition
        ``(state, action) → next_state`` are recomputed where possible, to
        avoid redundant computation over all states.

        Args:
            state: Current state (before the transition).
            action: Action taken.
            next_state: Resulting state.
        """
        s = self.state_to_idx[state]
        a = action
        s_prime = self.state_to_idx[next_state]

        if "novelty" in self.reward_types:
            cfg = self.reward_configs.get("novelty", {})
            if cfg:
                raise NotImplementedError(
                    "Novelty reward does not take any config at the moment, "
                    "are you sure you wanted to pass this config?"
                )

            # Novelty is relative (depends on all visit counts), so we must
            # recompute it for every state rather than just the current one.
            novelty_per_state = compute_novelty_for_all_states(self)  # shape: (|S|,)

            # Broadcast: novelty of next-state s' is the reward for any
            # transition (·, ·, s').
            self.raw_reward_components['novelty'][:, :, :] = novelty_per_state

            # Keep the global scaler up-to-date and re-scale the full array
            self.reward_scalers["novelty"].update(novelty_per_state)
            self.scaled_reward_components['novelty'] = self.reward_scalers["novelty"].scale(
                self.raw_reward_components['novelty']
            )

        if "empowerment" in self.reward_types:
            cfg = self.reward_configs.get("empowerment", {})
            learned_matrix, learned_state_to_idx = self.get_full_transition_matrix()

            num_steps_emp = cfg["num_steps"]
            if num_steps_emp != 1:
                raise NotImplementedError(
                    f"Empowerment for {num_steps_emp} steps requires changed "
                    "caching behavior, which is not implemented yet."
                )

            # Only recompute empowerment for the state whose row in the
            # transition matrix has just changed.
            changed_empowerment = compute_empowerment_for_state(
                state,
                transition_model=learned_matrix,
                state_to_idx=learned_state_to_idx,
                **cfg
            )

            # Cache: empowerment of next-state s' is the reward for any
            # transition (·, ·, s').
            self.raw_reward_components['empowerment'][:, :, s] = changed_empowerment

            self.reward_scalers["empowerment"].update(changed_empowerment)
            self.scaled_reward_components['empowerment'] = self.reward_scalers["empowerment"].scale(
                self.raw_reward_components['empowerment']
            )

        if "info_gain" in self.reward_types:
            cfg = self.reward_configs.get("info_gain", {})
            # Only recompute the (state, action) pair whose distribution changed
            changed_info_gain = calculate_predicted_information_gain_for_state_action_pair(
                state, action, self, **cfg
            )

            self.raw_reward_components['info_gain'][s, a] = changed_info_gain

            self.reward_scalers["info_gain"].update(changed_info_gain)
            self.scaled_reward_components['info_gain'] = self.reward_scalers["info_gain"].scale(
                self.raw_reward_components['info_gain']
            )

        # --- Combine reward components into the single R matrix ---
        if len(self.reward_types) == 0:
            # TODO: maybe compare "constant 1" here
            pass
        elif len(self.reward_types) == 1:
            self.R[:] = self.scaled_reward_components[self.reward_types[0]]
        elif len(self.reward_types) > 1:
            if self.combination_method == "mean":
                self.R[:] = np.mean(
                    [self.scaled_reward_components[r] for r in self.reward_types], axis=0
                )
            elif self.combination_method == "product":
                self.R[:] = np.prod(
                    [self.scaled_reward_components[r] for r in self.reward_types], axis=0
                )

    def _initialize_intrinsic_rewards(self) -> None:
        """Compute initial reward estimates under the current (prior) model.

        Called once during ``__init__`` (and again after injecting ground-truth
        counts in ``use_true_model`` mode).  Populates
        ``raw_reward_components``, ``scaled_reward_components``,
        ``reward_scalers``, and ``R``.
        """
        print("Updating rewards in first step")

        # Allocate per-reward storage
        self.reward_scalers: dict[str, GlobalRewardScaler] = {}
        self.raw_reward_components: dict[str, np.ndarray] = {}
        self.scaled_reward_components: dict[str, np.ndarray] = {}
        for reward_type in self.reward_types:
            if reward_type in ["novelty", "info_gain", "empowerment"]:
                self.reward_scalers[reward_type] = GlobalRewardScaler()
                self.raw_reward_components[reward_type] = np.zeros_like(self.R)
                self.scaled_reward_components[reward_type] = np.zeros_like(self.R)
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

        if "empowerment" in self.reward_types:
            cfg = self.reward_configs.get("empowerment", {})
            learned_matrix, learned_state_to_idx = self.get_full_transition_matrix()

            # Pass 1: compute raw empowerment for every state and update scaler
            for state in self.observed_states:
                state_idx = self.state_to_idx[state]
                empowerment_reward = compute_empowerment_for_state(
                    state,
                    transition_model=learned_matrix,
                    state_to_idx=learned_state_to_idx,
                    **cfg
                )
                self.reward_scalers["empowerment"].update(empowerment_reward)
                self.raw_reward_components["empowerment"][:, :, state_idx] = empowerment_reward

            # Pass 2: scale using the now-complete global bounds
            self.scaled_reward_components["empowerment"] = self.reward_scalers["empowerment"].scale(
                self.raw_reward_components["empowerment"]
            )

        if "info_gain" in self.reward_types:
            cfg = self.reward_configs.get("info_gain", {})
            # Pass 1: compute raw info-gain for every (state, action) pair
            for state in self.observed_states:
                state_idx = self.state_to_idx[state]
                for action in range(self.num_actions):
                    info_gain_reward = calculate_predicted_information_gain_for_state_action_pair(
                        state, action, self, **cfg
                    )
                    self.reward_scalers["info_gain"].update(info_gain_reward)
                    self.raw_reward_components["info_gain"][state_idx, action] = info_gain_reward

            # Pass 2: scale
            self.scaled_reward_components["info_gain"] = self.reward_scalers["info_gain"].scale(
                self.raw_reward_components["info_gain"]
            )

        if "novelty" in self.reward_types:
            cfg = self.reward_configs.get("novelty", {})
            # Pass 1: compute raw novelty for every state
            for state in self.observed_states:
                state_idx = self.state_to_idx[state]
                novelty_raw = compute_novelty_for_state(state_idx, self, **cfg)
                self.reward_scalers["novelty"].update(novelty_raw)
                self.raw_reward_components["novelty"][state_idx, :, :] = novelty_raw

            # Pass 2: scale
            self.scaled_reward_components["novelty"] = self.reward_scalers["novelty"].scale(
                self.raw_reward_components["novelty"]
            )

        # --- Combine reward components into the single R matrix ---
        if len(self.reward_types) == 0:
            # TODO: maybe compare "constant 1" here
            pass
        elif len(self.reward_types) == 1:
            self.R[:] = self.scaled_reward_components[self.reward_types[0]]
        elif len(self.reward_types) > 1:
            if self.combination_method == "mean":
                self.R[:] = np.mean(
                    [self.scaled_reward_components[r] for r in self.reward_types], axis=0
                )
            elif self.combination_method == "product":
                self.R[:] = np.prod(
                    [self.scaled_reward_components[r] for r in self.reward_types], axis=0
                )

        print("Reward components initialized in first step")
