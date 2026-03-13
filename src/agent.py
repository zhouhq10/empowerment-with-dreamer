"""Agent implementations for model-based intrinsic-motivation experiments.

Three concrete classes are provided:

- :class:`Agent` – abstract base class defining the ``select_action`` /
  ``update`` interface.
- :class:`RandomAgent` – selects actions uniformly at random (no learning).
- :class:`PrioritizedSweepingAgent` – maintains a :class:`CountBasedTransitionModel`
  and updates Q-values via a vectorised prioritized-sweeping loop after each
  environment step.

Adapted from:
    https://github.com/gruaz-lucas/Merits_of_curiosity/blob/main/src/agent.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.special import softmax

from transition_model import CountBasedTransitionModel


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class Agent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def select_action(self, state) -> int:
        """Select an action given the current state.

        Args:
            state: Current state.

        Returns:
            Selected action index.
        """
        pass

    @abstractmethod
    def update(self, state, action: int, next_state, reward: float, terminated: bool = False) -> None:
        """Update agent's internal state after a transition.

        Args:
            state: Current state.
            action: Taken action.
            next_state: Resulting state.
            reward: Received reward.
            terminated: Whether the episode has terminated.
        """
        pass


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Agent that selects actions uniformly at random."""

    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions

    def select_action(self, state) -> int:
        return np.random.choice(self.num_actions)

    def update(self, state, action: int, next_state, reward: float, terminated: bool = False) -> None:
        # Random agent doesn't need to learn anything
        pass


# ---------------------------------------------------------------------------
# Prioritized sweeping agent
# ---------------------------------------------------------------------------


class PrioritizedSweepingAgent(Agent):
    """Model-based agent with a count-based transition model and prioritized sweeping.

    After every environment transition the agent:
    1. Updates the transition model counts.
    2. Runs a vectorised prioritized-sweeping loop to propagate value changes.

    Action selection is *greedy* (ties broken randomly) so that the intrinsic
    reward structure drives exploration rather than arbitrary softmax noise.

    Args:
        num_actions: Number of discrete actions.
        all_states: Exhaustive list of environment states.
        gamma: Discount factor for the Bellman update.
        learning_rate: Step-size for Q-value updates (currently unused in
            vectorised sweep; kept for interface compatibility).
        temperature: Softmax temperature (currently unused; greedy selection
            is active).
        n_sweeps: Maximum number of prioritized-sweeping iterations per step.
            Defaults to ``100 * len(all_states)``.
        q_init: Initial Q-value for all (state, action) pairs.
        random_state: Seed for the internal RNG (tie-breaking, etc.).
        model_kwargs: Keyword arguments forwarded to
            :class:`CountBasedTransitionModel`.
    """

    def __init__(
        self,
        num_actions: int,
        all_states: list,
        gamma: float = 0.9,
        learning_rate: float = 1.0,
        temperature: float = 0.1,
        n_sweeps: int | None = None,
        q_init: float = 0.0,
        random_state: int = 42,
        model_kwargs: dict | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = learning_rate
        self.temperature = temperature
        self.q_init = q_init
        self.T_PS = n_sweeps if n_sweeps is not None else 100 * len(all_states)

        # State management
        self.state_to_idx: dict = {s: i for i, s in enumerate(all_states)}
        self.n_states = len(all_states)

        # Q-table and value function (U = max_a Q(s, a))
        self.q_table = np.full((self.n_states, num_actions), q_init)
        self.U = np.max(self.q_table, axis=1)

        # Learned transition model
        self.model = CountBasedTransitionModel(
            num_actions, states=all_states, **(model_kwargs or {})
        )

        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def select_action(self, state) -> int:
        """Return the greedy action for *state*, breaking ties randomly.

        Args:
            state: Current environment state.

        Returns:
            Action index with the highest Q-value (random among equals).
        """
        state_idx = self.state_to_idx[state]
        q_values = self.q_table[state_idx]

        # --- Greedy action selection ---
        # Softmax / temperature-scaled exploration was considered but greedy
        # selection is preferred: intrinsic rewards should drive exploration,
        # not arbitrary randomness.  Ties are broken uniformly at random.
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        action = self.rng.choice(best_actions)
        return action

    def update(self, state, action: int, next_state, reward: float, terminated: bool = False) -> None:
        """Update the transition model and run prioritized sweeping.

        Args:
            state: State before the transition.
            action: Action taken.
            next_state: Resulting state.
            reward: Extrinsic reward (passed to the model but not yet used).
            terminated: Whether the episode ended.
        """
        # Update the learned transition model (also refreshes intrinsic rewards)
        self.model.update(state, action, next_state, reward, terminated)

        # Propagate value changes to all affected states
        self.prioritized_sweeping()

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def prioritized_sweeping(self, ΔV_thresh: float = 1e-8, theta_thresh: float = 1e-8) -> None:
        """Vectorised prioritized sweeping over all states.

        Iterates up to ``T_PS`` times. Each iteration picks the state with the
        largest pending value change (|U[s] − V[s]|), commits the update, and
        propagates the delta to predecessor states weighted by the transition
        probability.  Terminates early when the relative value change falls
        below *ΔV_thresh*.

        The Q-update uses a scaled Bellman equation::

            Q(s, a) = Σ_{s'} T(s'|s,a) · [(1−λ)·R(s,a,s') + λ·U(s')·(1−terminal)]

        where ``λ = gamma`` and the ``(1−λ)`` factor keeps Q-values in [0, 1]
        when rewards are in [0, 1].

        Args:
            ΔV_thresh: Early-stopping threshold on relative value change.
            theta_thresh: Minimum T·|ΔV| to consider a predecessor state
                worth updating.
        """
        λ = self.gamma
        Q = self.q_table
        U = self.U
        R = self.model.R
        T, _ = self.model.get_full_transition_matrix()
        D = self.model.terminal_transitions  # True where transitions end the episode

        # Broadcast U(s') from shape (|S|,) to (|S|, |A|, |S|) for vectorised ops
        u_reshaped = np.tile(U, (self.n_states, self.num_actions, 1))

        # Scale R so Q-values stay in [0, 1] (rewards also in [0, 1])
        R_scaled = R * (1 - λ)
        r_plus_u = R_scaled + λ * u_reshaped * (~D)

        # Full Q update: Q(s,a) = Σ_{s'} T(s'|s,a)·[r(s,a,s') + λ·U(s')]
        Q[:] = np.sum(T * r_plus_u, axis=2)

        V = np.max(Q, axis=1)
        priorities = np.abs(U - V)

        for _ in range(self.T_PS):
            s_prime = np.argmax(priorities)
            ΔV = V[s_prime] - U[s_prime]

            # Early stopping: relative change is below threshold
            if np.max(V) != np.min(V) and abs(ΔV) / abs(np.max(V) - np.min(V)) <= ΔV_thresh:
                break

            U[s_prime] = V[s_prime]

            # Zero out priority to prevent immediate re-selection
            priorities[s_prime] = 0.0

            # Identify predecessor states worth updating
            # (those with T(s_prime | s, a) · |ΔV| > theta_thresh for at least one action)
            mask = T[:, :, s_prime] * np.abs(ΔV) > theta_thresh
            states_to_update = np.where(mask.any(axis=1))[0]

            if states_to_update.size > 0:
                # Incremental Q update for predecessors
                Q[states_to_update, :] = (
                    Q[states_to_update, :] + λ * T[states_to_update, :, s_prime] * ΔV
                )
                V[states_to_update] = np.max(Q[states_to_update, :], axis=1)
                priorities[states_to_update] = np.abs(U[states_to_update] - V[states_to_update])

        if _ == self.T_PS - 1:
            pass  # Skipping temporarily because the print statements clutter the output in the notebook, should probably log to logger instead
            # print("Warning: Prioritized sweeping did not converge within the specified number of iterations.")

        self.q_table = Q
        self.U = U

    def reset_model(self) -> None:
        raise NotImplementedError("Resetting the model is not implemented for this agent.")

        # Reset the value functions
        self.q_table.fill(self.q_init)
        self.U = np.max(self.q_table, axis=1)
