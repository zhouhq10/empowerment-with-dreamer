"""
Empowerment computation for tabular environments.

Empowerment of a state z is defined as the channel capacity between n-step action
sequences and the resulting next state:

    E(z) = max_{p(A|z)} I(A ; S' | S=z)

Three estimation strategies are provided:

* **count_unique_end_states_by_sampling** — Monte-Carlo, counts distinct end states.
* **approximate_next_state_entropy_by_sampling** — Monte-Carlo, estimates H(S'|S=z)
  under a uniform source policy.
* **blahut_arimoto** — Exact channel capacity via the Blahut-Arimoto algorithm; finds
  the optimal source policy rather than assuming it is uniform.
* **marginalize_over_uniform_policy** — Exact mutual information under a fixed uniform
  source policy (lower bound on true empowerment when the uniform policy is suboptimal).
"""

import itertools
from functools import reduce
from typing import Any, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt

from visualization import visualize_mutual_info_calculation
from utils import validate_probability_distribution

# Type alias used throughout for the empowerment-method argument.
EmpowermentMethod = Literal[
    "count_unique_end_states_by_sampling",
    "approximate_next_state_entropy_by_sampling",
    "blahut_arimoto",
    "marginalize_over_uniform_policy",
]
EnvType = Literal["minigrid", "gridworld"]


# ── Private helpers ────────────────────────────────────────────────────────────

def _rand_dist(shape: tuple[int, ...]) -> np.ndarray:
    """Return a random probability distribution of the given shape."""
    P = np.random.rand(*shape)
    return _normalize(P)


def _normalize(P: np.ndarray) -> np.ndarray:
    """Normalise an array to sum to 1 (in-place safe — returns a new array)."""
    s = sum(P)
    if s == 0.:
        raise ValueError("input distribution has sum zero")
    return P / s


# ── Sampling-based estimators ──────────────────────────────────────────────────

def calculate_empowerment_by_counting_unique_sampled_end_states(
    env: Any,
    obs: list,
    num_steps: int,
    num_samples: int,
) -> float:
    """Estimate empowerment by counting distinct end states reached under random action sequences.

    Samples `num_samples` random n-step action sequences, simulates each from `obs`,
    and returns log(|unique end states|).  This is a lower bound on true empowerment
    and assumes a uniform source policy.

    Args:
        env: GridWorld instance with attributes `num_actions` and mutable `agent_pos`.
        obs: Starting (x, y) position of the agent.
        num_steps: Length n of each action sequence.
        num_samples: Number of sequences to sample.

    Returns:
        Empowerment estimate in nats.
    """
    actions = list(range(env.num_actions))

    # All possible n-step action sequences (enumerated so we can sample by index)
    action_sequences = list(itertools.product(actions, repeat=num_steps))

    end_states: set = set()
    for _ in range(num_samples):
        sequence_idx = np.random.choice(len(action_sequences))
        sequence = action_sequences[sequence_idx]
        env.reset()
        env.agent_pos = obs.copy()

        for action in sequence:
            end_obs = env.step(action)

        end_states.add(tuple(end_obs))

    empowerment = np.log(len(end_states))
    return empowerment


def calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(
    env: Any,
    obs: list,
    num_steps: int,
    num_samples: int,
) -> float:
    """Estimate empowerment via the empirical entropy of the end-state distribution.

    Similar to :func:`calculate_empowerment_by_counting_unique_sampled_end_states`, but
    instead of counting unique end states it builds a histogram over end states and
    returns H(S'|S=obs) under the uniform source policy.  This is a lower bound on
    true empowerment.

    Args:
        env: GridWorld instance with attributes `num_actions`, `num_states`, and
             mutable `agent_pos`.
        obs: Starting (x, y) position of the agent.
        num_steps: Length n of each action sequence.
        num_samples: Number of sequences to sample.

    Returns:
        Empowerment estimate in nats.
    """
    actions = list(range(env.num_actions))

    # All possible n-step action sequences
    action_sequences = list(itertools.product(actions, repeat=num_steps))

    next_state_counts = np.zeros(env.num_states)
    for _ in range(num_samples):
        sequence_idx = np.random.choice(len(action_sequences))
        sequence = action_sequences[sequence_idx]
        env.reset()
        env.agent_pos = obs.copy()

        for action in sequence:
            env.step(action)

        next_state = env._get_state_index(env.agent_pos[0], env.agent_pos[1])
        next_state_counts[next_state] += 1

    next_state_probs = next_state_counts / next_state_counts.sum()
    empowerment = -np.sum(next_state_probs * np.log(next_state_probs + 1e-8))
    return empowerment


# ── Exact / model-based estimators ─────────────────────────────────────────────

def calculate_empowerment_under_uniform_policy(
    state: Any,
    p_next_state_given_state_action: np.ndarray,
    visualize: bool = False,
    state_names: Optional[list[str]] = None,
) -> float:
    """Compute I(A^n ; S' | S=state) under a fixed uniform source policy.

    This equals the true empowerment only when the uniform policy is optimal (e.g.
    when all actions are equally (de)terministic).  It is a lower bound otherwise.

    Args:
        state: Index of the current state (used only for visualisation).
        p_next_state_given_state_action: Transition probabilities
            p(S_{t+n} | A_t^n, S_t=state).  Shape ``(n_action_sequences, n_states)``,
            where each row is a distribution over next states for one action sequence.
        visualize: If True, call :func:`visualize_mutual_info_calculation` and show
            the plot before returning.
        state_names: Optional display names for states, forwarded to the visualiser.

    Returns:
        Mutual information I(A^n ; S_{t+n} | S_t=state) in bits.
    """
    # Uniform policy over all n-step action sequences: p(A^n | S=s)
    n_action_seqs = p_next_state_given_state_action.shape[0]
    p_action_given_state = np.ones(n_action_seqs) / n_action_seqs

    validate_probability_distribution(p_action_given_state)
    validate_probability_distribution(p_next_state_given_state_action, axis=1)

    # p(S_{t+n} | S_t=s) = sum_a p(S_{t+n} | a, s) * p(a | s)
    p_next_state_given_state = p_next_state_given_state_action.T @ p_action_given_state

    # H(S_{t+n} | S_t=s)
    H_marginal = -np.sum(p_next_state_given_state * np.log2(p_next_state_given_state + 1e-8))

    # H(S_{t+n} | A^n, S_t=s) = sum_a p(a|s) * H(S_{t+n} | a, s)
    H_conditional = 0
    for a in range(p_action_given_state.shape[0]):
        H_conditional += p_action_given_state[a] * -np.sum(
            p_next_state_given_state_action[a] * np.log2(p_next_state_given_state_action[a] + 1e-8)
        )

    # I(A^n ; S_{t+n} | S_t=s) = H(S_{t+n} | S_t=s) - H(S_{t+n} | A^n, S_t=s)
    mutual_information = H_marginal - H_conditional

    if visualize:
        visualize_mutual_info_calculation(
            current_state=state,
            p_next_state_given_state_action=p_next_state_given_state_action,
            p_action_given_state=p_action_given_state,
            state_names=state_names,
        )
        plt.show()

    return mutual_information


def blahut_arimoto_gopnik(
    P_yx: np.ndarray,
    state: Any,
    epsilon: float = 0.001,
    visualize_steps: bool = False,
    state_names: Optional[list[str]] = None,
) -> float:
    """Compute the channel capacity (= empowerment) via the Blahut-Arimoto algorithm.

    Finds the source policy p(A | S=z) that maximises the mutual information
    I(A ; S' | S=z), i.e. the true empowerment value:

        E(z) = max_{p(x)} I(x ; y)  where  x = A,  y = S'

    The algorithm iterates coordinate ascent on the (concave) mutual information
    objective and is guaranteed to converge to the global maximum regardless of the
    random initialisation.

    Implementation adapted from the repository of:
        Du, Y., Kosoy, E., Dayan, A., Rufova, M., Abbeel, P., & Gopnik, A. (2023).
        *What can AI Learn from Human Exploration? Intrinsically-Motivated Humans and
        Agents in Open-World Exploration.*  NeurIPS 2023 Workshop on Information-
        Theoretic Principles in Cognitive Systems.
    Original source: https://github.com/Mchristos/empowerment

    Args:
        P_yx: Channel matrix p(y | x) = p(S' | A, S=z).
              Shape ``(n_states, n_actions)`` — each *column* is a distribution over
              next states for one action.
        state: Index of the current state (used only for visualisation).
        epsilon: Convergence threshold on the duality gap T.
        visualize_steps: If True, visualise the mutual-information decomposition every
            10 iterations and at convergence.
        state_names: Optional display names for states, forwarded to the visualiser.

    Returns:
        Channel capacity C = E(z) in bits.  Clamped to 0 if numerical error yields
        a slightly negative value.
    """
    eps = 1e-40
    P_yx = P_yx + eps  # p(S' | A, S=z) — add small floor to avoid log(0)
    P_yx = P_yx / np.sum(P_yx, axis=0)  # renormalise columns after floor

    # P_yx is (|S|, |A|): each column must sum to 1
    validate_probability_distribution(P_yx, axis=0)

    # Initialise source policy q(a) randomly
    q_x = _rand_dist((P_yx.shape[1],))
    T = 1  # duality gap (initialised above epsilon to enter the loop)

    iteration = 0
    while T > epsilon:
        # E-step: posterior p(A | S', S=z) = p(S'|A,z) * q(A) / p(S'|z)
        PHI_yx = (P_yx * q_x.reshape(1, -1)) / (P_yx @ q_x).reshape(-1, 1)

        # M-step: unnormalised updated policy
        r_x = np.exp(np.sum(P_yx * np.log2(PHI_yx), axis=0))

        # Channel capacity estimate and duality gap
        C = np.log2(np.sum(r_x))
        T = np.max(np.log2(r_x / q_x)) - C

        # Update source policy
        q_x = _normalize(r_x + eps)

        if visualize_steps and (iteration % 10 == 0 or T <= epsilon):
            if T <= epsilon:
                print(f"Converged after {iteration} iterations, T={T}, C={C}")
                print("Final source policy and next state distribution:")
            else:
                print(f"Iteration {iteration}, T={T}, C={C}")

            # P_yx is (|S|, |A|) but visualiser expects (|A|, |S|)
            visualize_mutual_info_calculation(
                current_state=state,
                p_next_state_given_state_action=P_yx.T,
                p_action_given_state=q_x,
                state_names=state_names,
            )
            plt.show()

        iteration += 1

    if C < 0:
        C = 0
    return C


# ── Batch helpers ──────────────────────────────────────────────────────────────

def _build_nstep_transition_tensor(
    T: np.ndarray,
    num_steps: int,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Pre-compute the n-step transition tensor T_n for all action sequences.

    Args:
        T: One-step transition matrix of shape ``(n_states, n_actions, n_states)``.
        num_steps: Horizon n.

    Returns:
        nstep_action_samples: List of all n-step action sequences (tuples of ints).
        T_n: Tensor of shape ``(n_states, n_action_sequences, n_states)`` where
             ``T_n[s, i, s']`` = p(S_{t+n} = s' | A_t^n = seq_i, S_t = s).
    """
    n_states = T.shape[0]
    nstep_action_samples = list(itertools.product(range(T.shape[1]), repeat=num_steps))

    T_n = np.zeros([n_states, len(nstep_action_samples), n_states])
    for i, an in enumerate(nstep_action_samples):
        # Chain single-step matrices: T[:,a_n,:] @ ... @ T[:,a_1,:]
        T_n[:, i, :] = reduce(lambda x, y: np.dot(y, x), map(lambda a: T[:, a, :], an))

    return nstep_action_samples, T_n


def compute_empowerment_for_all_states(
    num_steps: int,
    method: EmpowermentMethod,
    env: Any,
    num_samples: Optional[int] = None,
    visualize_mutual_info_decomposition: bool = False,
    states_to_visualize: list[tuple] = [(0, 0)],
    transition_model: Optional[np.ndarray] = None,
    state_to_idx: Optional[dict] = None,
    action_to_idx: Optional[dict] = None,
    env_type: EnvType = "minigrid",
) -> np.ndarray:
    """Compute empowerment for every state in the environment.

    Args:
        num_steps: Horizon n — length of the action sequences considered.
        method: Estimation method.  One of ``"count_unique_end_states_by_sampling"``,
            ``"approximate_next_state_entropy_by_sampling"``, ``"blahut_arimoto"``,
            ``"marginalize_over_uniform_policy"``.
        env: Environment instance (GridWorld or wrapped MiniGridEnv).
        num_samples: Required for sampling-based methods; ignored otherwise.
        visualize_mutual_info_decomposition: If True, visualise the MI decomposition
            for states in `states_to_visualize`.
        states_to_visualize: (x, y) coordinates for which to show the MI breakdown.
        transition_model: Estimated transition probabilities of shape
            ``(n_states, n_actions, n_states)`` to use instead of the environment's
            true dynamics.  Required for model-based methods in the MiniGrid case.
        state_to_idx: Mapping from state tuples to row indices in `transition_model`.
            Required when `transition_model` is provided.
        action_to_idx: Unused; reserved for future use.
        env_type: ``"minigrid"`` or ``"gridworld"``.

    Returns:
        empowerment_map: Array of empowerment values indexed by state coordinates.
            Shape ``(width, height, 4)`` for MiniGrid (last axis = orientation) or
            ``(width, height)`` for GridWorld.  Obstacle / invalid states are NaN.
    """
    if env_type == "minigrid":
        if transition_model is not None and state_to_idx is not None:
            validate_probability_distribution(transition_model, axis=2)
            idx_to_state = {v: k for k, v in state_to_idx.items()}
            num_states = len(state_to_idx)
        else:
            raise NotImplementedError

        empowerment_map = np.full((env.unwrapped.width, env.unwrapped.height, 4), np.nan)

        T = transition_model
        nstep_action_samples, T_n = _build_nstep_transition_tensor(T, num_steps)

        for s_idx in range(num_states):
            s = idx_to_state[s_idx]
            x, y, dir = s
            visualize = visualize_mutual_info_decomposition and (x, y) in states_to_visualize

            if method == "count_unique_end_states_by_sampling":
                if num_samples is None:
                    raise ValueError("num_samples must be specified for method 'count_unique_end_states_by_sampling'")
                empowerment_map[x, y, dir] = calculate_empowerment_by_counting_unique_sampled_end_states(
                    env, [x, y], num_steps, num_samples
                )
            elif method == "approximate_next_state_entropy_by_sampling":
                if num_samples is None:
                    raise ValueError("num_samples must be specified for method 'approximate_next_state_entropy_by_sampling'")
                empowerment_map[x, y, dir] = calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(
                    env, [x, y], num_steps, num_samples
                )
            elif method == "blahut_arimoto":
                # T_n[s_idx,:,:] is (n_action_seqs, n_states); BA expects (n_states, n_action_seqs)
                empowerment_map[x, y, dir] = blahut_arimoto_gopnik(
                    T_n[s_idx, :, :].T,
                    state=s_idx,
                    epsilon=1e-4,
                    visualize_steps=visualize,
                    state_names=[str(idx_to_state[i]) for i in range(num_states)],
                )
            elif method == "marginalize_over_uniform_policy":
                empowerment_map[x, y, dir] = calculate_empowerment_under_uniform_policy(
                    s_idx, T_n[s_idx, :, :],
                    visualize=visualize,
                    state_names=[str(idx_to_state[i]) for i in range(num_states)],
                )
            else:
                raise ValueError(f"Unknown method: {method}")

    elif env_type == "gridworld":
        if transition_model is not None and state_to_idx is not None:
            validate_probability_distribution(transition_model, axis=2)
            idx_to_state = {v: k for k, v in state_to_idx.items()}
            num_states = len(state_to_idx)
        else:
            idx_to_state = {env._get_state_index(x, y): (x, y)
                            for x in range(env.width) for y in range(env.height)}
            num_states = env.num_states

        empowerment_map = np.full((env.width, env.height), np.nan)

        T = transition_model if transition_model is not None else env.transition_prob
        nstep_action_samples, T_n = _build_nstep_transition_tensor(T, num_steps)

        for s_idx in range(num_states):
            s = idx_to_state[s_idx]
            x, y = s

            if (x, y) in env.obstacles:
                # Leave as NaN — agent can never occupy obstacle cells
                empowerment_map[x, y] = np.nan
                continue

            visualize = visualize_mutual_info_decomposition and (x, y) in states_to_visualize

            if method == "count_unique_end_states_by_sampling":
                if num_samples is None:
                    raise ValueError("num_samples must be specified for method 'count_unique_end_states_by_sampling'")
                empowerment_map[x, y] = calculate_empowerment_by_counting_unique_sampled_end_states(
                    env, [x, y], num_steps, num_samples
                )
            elif method == "approximate_next_state_entropy_by_sampling":
                if num_samples is None:
                    raise ValueError("num_samples must be specified for method 'approximate_next_state_entropy_by_sampling'")
                empowerment_map[x, y] = calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(
                    env, [x, y], num_steps, num_samples
                )
            elif method == "blahut_arimoto":
                # T_n[s_idx,:,:] is (n_action_seqs, n_states); BA expects (n_states, n_action_seqs)
                empowerment_map[x, y] = blahut_arimoto_gopnik(
                    T_n[s_idx, :, :].T,
                    state=s_idx,
                    epsilon=1e-4,
                    visualize_steps=visualize,
                    state_names=[str(idx_to_state[i]) for i in range(num_states)],
                )
            elif method == "marginalize_over_uniform_policy":
                empowerment_map[x, y] = calculate_empowerment_under_uniform_policy(
                    s_idx, T_n[s_idx, :, :],
                    visualize=visualize,
                    state_names=[str(idx_to_state[i]) for i in range(num_states)],
                )
            else:
                raise ValueError(f"Unknown method: {method}")
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented.")

    return empowerment_map


def compute_empowerment_for_state(
    state: Any,
    num_steps: int,
    method: EmpowermentMethod,
    num_samples: Optional[int] = None,
    visualize_mutual_info_decomposition: bool = False,
    states_to_visualize: list[tuple] = [(0, 0)],
    transition_model: Optional[np.ndarray] = None,
    state_to_idx: Optional[dict] = None,
    env_type: EnvType = "minigrid",
) -> float:
    """Compute empowerment for a single state.

    Identical to :func:`compute_empowerment_for_all_states` but for one state only,
    avoiding the overhead of iterating over the full state space.  Used during online
    learning to update only the state whose model changed.

    Args:
        state: State tuple — ``(x, y, dir)`` for MiniGrid, ``(x, y)`` for GridWorld.
        num_steps: Horizon n.
        method: Estimation method (see :func:`compute_empowerment_for_all_states`).
        num_samples: Required for sampling-based methods.
        visualize_mutual_info_decomposition: If True, visualise the MI decomposition
            when `state` is in `states_to_visualize`.
        states_to_visualize: States for which to show the MI breakdown.
        transition_model: Estimated transition probabilities, shape
            ``(n_states, n_actions, n_states)``.
        state_to_idx: Mapping from state tuples to row indices in `transition_model`.
        env_type: ``"minigrid"`` or ``"gridworld"``.

    Returns:
        Empowerment value for `state` in bits.
    """
    if env_type == "minigrid":
        if transition_model is not None and state_to_idx is not None:
            validate_probability_distribution(transition_model, axis=2)
            idx_to_state = {v: k for k, v in state_to_idx.items()}
            num_states = len(state_to_idx)
        else:
            raise NotImplementedError

        T = transition_model
        _, T_n = _build_nstep_transition_tensor(T, num_steps)

        empowerment = 0
        x, y, dir = state
        s_idx = state_to_idx[state]
        visualize = visualize_mutual_info_decomposition and (x, y, dir) in states_to_visualize

        if method == "count_unique_end_states_by_sampling":
            if num_samples is None:
                raise ValueError("num_samples must be specified for method 'count_unique_end_states_by_sampling'")
            empowerment = calculate_empowerment_by_counting_unique_sampled_end_states(
                env, [x, y], num_steps, num_samples  # type: ignore[name-defined]
            )
        elif method == "approximate_next_state_entropy_by_sampling":
            if num_samples is None:
                raise ValueError("num_samples must be specified for method 'approximate_next_state_entropy_by_sampling'")
            empowerment = calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(
                env, [x, y], num_steps, num_samples  # type: ignore[name-defined]
            )
        elif method == "blahut_arimoto":
            empowerment = blahut_arimoto_gopnik(
                T_n[s_idx, :, :].T,
                state=s_idx,
                epsilon=1e-4,
                visualize_steps=visualize,
            )
        elif method == "marginalize_over_uniform_policy":
            empowerment = calculate_empowerment_under_uniform_policy(
                s_idx, T_n[s_idx, :, :], visualize=visualize
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    elif env_type == "gridworld":
        if transition_model is not None and state_to_idx is not None:
            validate_probability_distribution(transition_model, axis=2)
            idx_to_state = {v: k for k, v in state_to_idx.items()}
            num_states = len(state_to_idx)
        else:
            idx_to_state = {env._get_state_index(x, y): (x, y)  # type: ignore[name-defined]
                            for x in range(env.width) for y in range(env.height)}
            num_states = env.num_states  # type: ignore[name-defined]

        T = transition_model if transition_model is not None else env.transition_prob  # type: ignore[name-defined]
        _, T_n = _build_nstep_transition_tensor(T, num_steps)

        empowerment = 0
        x, y = state
        s_idx = state_to_idx[state] if state_to_idx is not None else env._get_state_index(x, y)  # type: ignore[name-defined]

        if (x, y) in env.obstacles:  # type: ignore[name-defined]
            raise ValueError("Cannot compute empowerment for an obstacle state.")

        visualize = visualize_mutual_info_decomposition and (x, y) in states_to_visualize

        if method == "count_unique_end_states_by_sampling":
            if num_samples is None:
                raise ValueError("num_samples must be specified for method 'count_unique_end_states_by_sampling'")
            empowerment = calculate_empowerment_by_counting_unique_sampled_end_states(
                env, [x, y], num_steps, num_samples  # type: ignore[name-defined]
            )
        elif method == "approximate_next_state_entropy_by_sampling":
            if num_samples is None:
                raise ValueError("num_samples must be specified for method 'approximate_next_state_entropy_by_sampling'")
            empowerment = calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(
                env, [x, y], num_steps, num_samples  # type: ignore[name-defined]
            )
        elif method == "blahut_arimoto":
            empowerment = blahut_arimoto_gopnik(
                T_n[s_idx, :, :].T,
                state=s_idx,
                epsilon=1e-4,
                visualize_steps=visualize,
            )
        elif method == "marginalize_over_uniform_policy":
            empowerment = calculate_empowerment_under_uniform_policy(
                s_idx, T_n[s_idx, :, :], visualize=visualize
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented.")

    return empowerment
