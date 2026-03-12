"""Utilities for running agent experiments and caching results to disk.

Two public functions are provided:

- :func:`run_agent` – run an agent for a fixed number of steps, collecting
  detailed history (Q-values, reward estimates, transitions, etc.) at regular
  evaluation checkpoints.
- :func:`run_or_load` – thin wrapper around :func:`run_agent` that
  serialises the returned history to a pickle file so that the same
  experiment is never run twice.
"""

from __future__ import annotations

import os
from copy import deepcopy
import random

import dill as pickle
from tqdm import tqdm
import numpy as np

from environment import get_all_states, get_ground_truth_transition_probabilities
from info_gain import calculate_predicted_information_gain_for_state_action_pair, compute_information_gain_for_all_states
from intrinsic.empowerment import compute_empowerment_for_all_states, compute_empowerment_for_state
from novelty import compute_novelty_for_state
from profiler import create_profiler


def run_agent(
    env,
    agent,
    n_steps: int,
    rewards: list[str] = ["info_gain", "empowerment"],
    combination_method: str = "mean",
    eval_interval: int = 100,
    known_state_space: bool = True,
    use_true_model: bool = False,
    seed: int = 31571939,
) -> dict:
    """Run *agent* in *env* for *n_steps* and record evaluation snapshots.

    At every *eval_interval*-th step the function records:
    * Number of discovered states.
    * Deep copies of the transition model.
    * Q-values and expected rewards per state (for ``PrioritizedSweepingAgent``).
    * Predicted information-gain, empowerment, and novelty heatmaps.

    Args:
        env: GridWorld (or wrapped MiniGrid) environment.
        agent: Agent instance with ``select_action`` and ``update`` methods.
        n_steps: Total number of environment steps.
        rewards: Reward types tracked in the history.  Any subset of
            ``["extrinsic", "info_gain", "empowerment", "novelty"]``.
        combination_method: ``"mean"`` or ``"product"`` (only used for labelling).
        eval_interval: Number of steps between evaluation checkpoints.
        known_state_space: Must be ``True`` (unknown state space is not
            implemented).
        use_true_model: If ``True``, inject the analytically correct
            transition counts before the run so the agent plans with the
            ground-truth model.
        seed: Random seed for reproducibility.

    Returns:
        history: Dictionary with the following keys:

        * ``n_states`` – list of state-count snapshots.
        * ``eval_steps`` / ``eval_episodes`` – step / episode at each checkpoint.
        * ``transition_model`` – deep-copied model at each checkpoint.
        * ``q_values_per_state`` – Q-table reshaped to grid coordinates.
        * ``agent_rewards`` – mean reward per (state, action) on the grid.
        * ``agent_rewards_through_model`` – expected reward under the model.
        * ``info_gain_estimates`` – predicted information-gain heatmap.
        * ``empowerment_estimates`` – empowerment heatmap.
        * ``novelty_estimates`` – novelty heatmap.
        * ``transitions`` – raw ``(s, a, s', r, terminated, truncated)`` list.
        * ``raw_rewards`` / ``scaled_rewards`` – per-reward-type snapshots.
        * ``agent_rewards`` – combined reward used by the agent.
        * ``final_agent`` – deep copy of the agent at the end of the run.
    """

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    history: dict = {
        'n_states': [],
        'eval_steps': [],
        'eval_episodes': [],
        'empowerment_estimates': [],
        'q_values_per_state': [],
        'transition_model': [],
        'info_gain_estimates': [],
        'novelty_estimates': [],
        'transitions': [],
        'raw_rewards': {reward_type: [] for reward_type in rewards},
        'scaled_rewards': {reward_type: [] for reward_type in rewards},
        'agent_rewards': [],
        'agent_rewards_through_model': [],
        'agent_R': [],
    }

    if not known_state_space:
        raise NotImplementedError("Unknown state space not implemented.")

    if use_true_model:
        print("Initializing model counts with ground truth probabilities.")
        P_true, terminal_transitions, _ = get_ground_truth_transition_probabilities(env.unwrapped)

        # Convert ground-truth probabilities to high-confidence pseudo-counts
        # so that the Dirichlet posterior is sharply concentrated on the truth.
        agent.model.counts = (P_true * 1_000_000.0) + agent.model.prior_count
        agent.model.terminal_transitions = terminal_transitions

        # Reinitialise intrinsic rewards under the true transition model
        agent.model._initialize_intrinsic_rewards()
        agent.model.is_true_model = True

    # Pre-compute evaluation checkpoints
    eval_steps = range(0, n_steps + 1, eval_interval)
    next_eval = eval_steps[0]
    eval_idx = 1
    episode = 0

    state, _ = env.reset()

    for step in tqdm(range(n_steps)):
        # --- Interact with the environment ---
        action = agent.select_action(state)
        next_state, extrinsic_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if "extrinsic" in rewards:
            # Scale extrinsic reward to [0, 1]
            raise NotImplementedError("Extrinsic reward not implemented.")
            scaled_extrinsic_reward = extrinsic_reward_scaler.scale(extrinsic_reward)
            reward_dict["extrinsic"] = scaled_extrinsic_reward

        # Update agent (model counts + prioritized sweeping)
        agent.update(state, action, next_state, extrinsic_reward, terminated)

        history['transitions'].append((state, action, next_state, extrinsic_reward, terminated, truncated))

        # --- Evaluation checkpoint ---
        if step == next_eval:
            history['n_states'].append(len(agent.model.observed_states))
            history['eval_steps'].append(step)
            history['eval_episodes'].append(episode)
            history['transition_model'].append(deepcopy(agent.model))

            if agent.__class__.__name__ in ('QAgent', 'PrioritizedSweepingAgent'):
                # Q-values reshaped to (width, height, 4_dirs, num_actions)
                q_values = np.full(
                    (env.unwrapped.width, env.unwrapped.height, 4, agent.num_actions), np.nan
                )
                for s in agent.model.observed_states:
                    x, y, dir = s
                    q_values[x, y, dir] = agent.q_table[agent.state_to_idx[s]]
                history['q_values_per_state'].append(q_values)

                # Mean reward per (state, action) – averaged over next-states
                agent_rewards = np.full(
                    (env.unwrapped.width, env.unwrapped.height, 4, agent.num_actions), np.nan
                )
                for s in agent.model.observed_states:
                    x, y, dir = s
                    agent_rewards[x, y, dir] = agent.model.R[agent.state_to_idx[s]].mean(axis=1)
                history['agent_rewards'].append(agent_rewards)

                # Expected reward under the model: Σ_{s'} p(s'|s,a) · R(s,a,s')
                agent_rewards_through_model = np.full(
                    (env.unwrapped.width, env.unwrapped.height, 4, agent.num_actions), np.nan
                )
                learned_matrix, learned_state_to_idx = agent.model.get_full_transition_matrix()
                for s in agent.model.observed_states:
                    x, y, dir = s
                    state_idx = agent.state_to_idx[s]
                    for a in range(agent.num_actions):
                        agent_rewards_through_model[x, y, dir, a] = np.dot(
                            learned_matrix[state_idx, a], agent.model.R[state_idx, a]
                        )
                history['agent_rewards_through_model'].append(agent_rewards_through_model)

            # Compute and store predicted information gain
            cfg = agent.model.reward_configs.get("info_gain", {})
            predicted_info_gain = compute_information_gain_for_all_states(env, agent.model, **cfg)
            history['info_gain_estimates'].append(predicted_info_gain)

            # Compute and store empowerment estimates
            cfg = agent.model.reward_configs.get("empowerment", {})
            learned_matrix, learned_state_to_idx = agent.model.get_full_transition_matrix()
            empowerment_estimates = compute_empowerment_for_all_states(
                env=env,
                transition_model=learned_matrix,
                state_to_idx=learned_state_to_idx,
                **cfg
            )
            history['empowerment_estimates'].append(empowerment_estimates)

            # Compute and store novelty estimates
            cfg = agent.model.reward_configs.get("novelty", {})
            novelty_estimates = np.full(
                (env.unwrapped.width, env.unwrapped.height, 4), np.nan
            )
            for s in agent.model.observed_states:
                x, y, dir = s
                novelty_estimates[x, y, dir] = compute_novelty_for_state(
                    agent.model.state_to_idx[s],
                    agent.model,
                    **cfg
                )
            history['novelty_estimates'].append(novelty_estimates)

            if eval_idx < len(eval_steps):
                next_eval = eval_steps[eval_idx]
                eval_idx += 1

        state = next_state

        # Reset environment after episode termination (but not truncation)
        if terminated:
            episode += 1
            state, _ = env.reset()

    # Store the final agent (includes the fully trained model)
    history['final_agent'] = deepcopy(agent)

    return history


def run_or_load(
    env,
    agent,
    rewards: list[str],
    n_steps: int,
    eval_interval: int,
    combination_method: str | None = None,
    use_true_model: bool = False,
    seed: int = 31571939,
    custom_identifier: str | None = None,
    path: str = "/mnt/lustre/work/wu/wkn758/empowerment-and-human-behavior/dat/toy_env_runs",
) -> dict:
    """Run the agent (or reload from disk if the result already exists).

    The run is identified by a filename that encodes all relevant
    hyperparameters.  If the corresponding pickle file exists on *path*, it
    is loaded and returned directly; otherwise :func:`run_agent` is called
    and the result is saved.

    Args:
        env: Environment instance.
        agent: Agent instance.
        rewards: List of reward types (used to construct the filename).
        n_steps: Total number of steps.
        eval_interval: Evaluation interval.
        combination_method: Reward combination method (required for multiple
            rewards).
        use_true_model: Whether to use the ground-truth transition model.
        seed: Random seed.
        custom_identifier: Optional free-form string appended to the filename.
        path: Directory where pickle files are read from / written to.

    Returns:
        history dict as returned by :func:`run_agent`.
    """
    agent_name = agent.__class__.__name__

    # Build a human-readable identifier for this run configuration
    run_str = f"{agent_name}"
    if agent_name != "RandomAgent":
        if len(rewards) > 0:
            run_str += f"-reward_{'_'.join(rewards)}"
        else:
            run_str += "-reward_none"
        if len(rewards) > 1:
            if combination_method is None:
                raise ValueError("Combination method must be specified for multiple rewards.")
            run_str += f"-{combination_method}"

    if agent_name in ("QAgent", "PrioritizedSweepingAgent"):
        run_str += f"-gamma_{agent.gamma}-lr_{agent.lr}-qinit_{agent.q_init}"

    run_str += f"-nsteps_{n_steps}-eval_{eval_interval}-seed_{seed}"

    if use_true_model:
        run_str += "-true_model"

    if rewards is not None and "empowerment" in rewards:
        empowerment_cfg = agent.model.reward_configs.get("empowerment", {})
        run_str += f"-emp_steps_{empowerment_cfg['num_steps']}"

    if custom_identifier is not None:
        run_str += f"-{custom_identifier}"

    filename = os.path.join(path, f"{run_str}.pkl")

    try:
        with open(filename, 'rb') as f:
            history = pickle.load(f)
        print(f"Run exists: {filename}, loaded history from file.")
    except FileNotFoundError:
        print(f"Run does not exist: {filename}, running agent...")
        history = run_agent(
            env, agent, n_steps, rewards, combination_method, eval_interval,
            seed=seed, use_true_model=use_true_model
        )
        with open(filename, 'wb') as f:
            pickle.dump(history, f)

    return history
