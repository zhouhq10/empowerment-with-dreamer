import os
from copy import deepcopy
import random

import dill as pickle
from tqdm import tqdm
import numpy as np

from environment import get_all_states, get_ground_truth_transition_probabilities
from info_gain import calculate_predicted_information_gain_for_state_action_pair, compute_information_gain_for_all_states
from empowerment import compute_empowerment_for_all_states, compute_empowerment_for_state
from novelty import compute_novelty_for_state
from profiler import create_profiler

def run_agent(env, agent, n_steps, rewards=["info_gain", "empowerment"], combination_method="mean", eval_interval=100, known_state_space=True, use_true_model=False, seed=31571939):
    """Run any agent and track transition model learning and intrinsic motivations.
    
    Args:
        env: GridWorld environment
        agent: Instance of Agent class
        n_steps: Number of steps to run
        rewards: List of rewards for the agent to receive. Can be "extrinsic", "info_gain", and/or "empowerment".
        combination_method: Method to combine multiple rewards. Can be "mean" or "product".
        eval_interval: Evaluate transition model every eval_interval steps
        known_state_space: If True, use known state space. If False, use unknown state space.
        use_true_model: If True, use the true model of the environment instead of the learned model (only for empowerment).
        seed: Random seed
        
    Returns:
        history: Dictionary containing:
            - transition_model: Final learned model
            - n_states: List of number of discovered states
            - eval_steps: Steps at which evaluations were performed
            - eval_episodes: Episodes at which evaluations were performed
            - empowerment_estimates: List of empowerment estimates at different timesteps
            - q_values: List of Q-values at different timesteps
            - transition_model: List of transition models at different timesteps
            - info_gain_estimates: List of predicted information gain estimates at different timesteps
    """

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)

    history = {
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

        # Update the model with the true values
        # Since we use counts, multiply the probabilities by a large value to approximate high confidence counts
        agent.model.counts = (P_true * 1_000_000.0) + agent.model.prior_count 
        agent.model.terminal_transitions = terminal_transitions

        # Reinitialize the intrinsic rewards with the true probabilities
        agent.model._initialize_intrinsic_rewards()
        agent.model.is_true_model = True
        
    # Set up evaluation steps
    eval_steps = range(0, n_steps + 1, eval_interval)
    
    next_eval = eval_steps[0]
    eval_idx = 1
    episode = 0
    
    state, _ = env.reset()
    
    for step in tqdm(range(n_steps)):
        # Agent selects action
        action = agent.select_action(state)
        
        # Take step
        next_state, extrinsic_reward, terminated, truncated, _ = env.step(action)
        
        done = terminated or truncated
        
        if "extrinsic" in rewards:
            # Scale extrinsic reward to [0, 1]
            raise NotImplementedError("Extrinsic reward not implemented.")
            scaled_extrinsic_reward = extrinsic_reward_scaler.scale(extrinsic_reward)
            reward_dict["extrinsic"] = scaled_extrinsic_reward
        
        agent.update(state, action, next_state, extrinsic_reward, terminated) 

        # Store transition
        history['transitions'].append((state, action, next_state, extrinsic_reward, terminated, truncated))

        if step == next_eval:
            history['n_states'].append(len(agent.model.observed_states))
            history['eval_steps'].append(step)
            history['eval_episodes'].append(episode)
            history['transition_model'].append(deepcopy(agent.model))
            
            if agent.__class__.__name__ == 'QAgent' or agent.__class__.__name__ == 'PrioritizedSweepingAgent':
                # Store q-values for later analysis
                q_values = np.full((env.unwrapped.width, env.unwrapped.height, 4, agent.num_actions), np.nan)
                for s in agent.model.observed_states:
                    x, y, dir = s
                    q_values[x, y, dir] = agent.q_table[agent.state_to_idx[s]]
                # Report entire q-value array including directions to allow for more detailed analysis
                history['q_values_per_state'].append(q_values)

                # store agent rewards for later analysis
                agent_rewards = np.full((env.unwrapped.width, env.unwrapped.height, 4, agent.num_actions), np.nan)
                for s in agent.model.observed_states:
                    x, y, dir = s
                    agent_rewards[x, y, dir] = agent.model.R[agent.state_to_idx[s]].mean(axis=1)
                history['agent_rewards'].append(agent_rewards)

                # also store the expected rewards under the model
                agent_rewards_through_model = np.full((env.unwrapped.width, env.unwrapped.height, 4, agent.num_actions), np.nan)
                learned_matrix, learned_state_to_idx = agent.model.get_full_transition_matrix()
                for s in agent.model.observed_states:
                    x, y, dir = s
                    state_idx = agent.state_to_idx[s]
                    for a in range(agent.num_actions):
                        # compute reward as p(s'|s, a) * R(s, a, s')
                        agent_rewards_through_model[x, y, dir, a] = np.dot(learned_matrix[state_idx, a], agent.model.R[state_idx, a])
                history['agent_rewards_through_model'].append(agent_rewards_through_model)
                    
                # store agent R for later analysis
                # history['agent_R'].append(agent.model.R.copy())

            
            # Compute predicted information gain 
            cfg = agent.model.reward_configs.get("info_gain", {})
            predicted_info_gain = compute_information_gain_for_all_states(env, agent.model, **cfg)
            history['info_gain_estimates'].append(predicted_info_gain)
            
            # Compute empowerment based on transition model estimate
            cfg = agent.model.reward_configs.get("empowerment", {})
            learned_matrix, learned_state_to_idx = agent.model.get_full_transition_matrix()
            empowerment_estimates = compute_empowerment_for_all_states(
                env=env,
                transition_model=learned_matrix, 
                state_to_idx=learned_state_to_idx, 
                **cfg
                )
            history['empowerment_estimates'].append(empowerment_estimates)
            
            # Store novelty estimates
            cfg = agent.model.reward_configs.get("novelty", {})
            novelty_estimates = np.full((env.unwrapped.width, env.unwrapped.height, 4), np.nan)
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
        
        # Reset environment if dead, but not if truncated
        if terminated:
            episode += 1
            state, _ = env.reset()

    # At the end of the run, store the entire agent object (which includes the learned model)
    history['final_agent'] = deepcopy(agent)
    
    return history


def run_or_load(env, agent, rewards, n_steps, eval_interval, combination_method=None, use_true_model=False, seed=31571939, custom_identifier=None, path="/mnt/lustre/work/wu/wkn758/empowerment-and-human-behavior/dat/toy_env_runs"):
    """Run agent and save history to disk if it doesn't already exist."""
    agent_name = agent.__class__.__name__
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
        
    if agent_name == "QAgent" or agent_name == "PrioritizedSweepingAgent":
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
        history = run_agent(env, agent, n_steps, rewards, combination_method, eval_interval, seed=seed, use_true_model=use_true_model)
        
        # save history to disk
        with open(filename, 'wb') as f:
            pickle.dump(history, f)
    
    return history