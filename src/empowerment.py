import itertools
from functools import reduce
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from visualization import visualize_mutual_info_calculation
from utils import validate_probability_distribution


def calculate_empowerment_by_counting_unique_sampled_end_states(env, obs, num_steps, num_samples):
    """ Calculate empowerment of state 'obs' by counting unique end states of random action sequences starting in 'obs'.
    
    Args:
        env: GridWorld object
        obs: initial observation
        num_steps: number of steps in each action sequence
        num_samples: number of random action sequences to sample
        
    Returns:
        empowerment: empowerment value    
    """
    
    actions = list(range(env.num_actions))
    
    # Generate all possible action sequences
    action_sequences = list(itertools.product(actions, repeat=num_steps))
    
    end_states = set()
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

def calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(env, obs, num_steps, num_samples):
    """ Calculate empowerment of state 'obs' by approximating the entropy of the distribution of next states through sampling. 
    
    Similar to calculate_empowerment_by_counting_unique_sampled_end_states, but instead of counting unique end states,
    we approximate the distribution of next states by sampling, and then calculate the empowerment based on the entropy
    of this distribution.
    
    Args:
        env: GridWorld object
        obs: initial observation
        num_steps: number of steps in each action sequence
        num_samples: number of random action sequences to sample
        
    Returns:
        empowerment: empowerment value
        
    """
        
    actions = list(range(env.num_actions))
    
    # Generate all possible action sequences
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


def calculate_empowerment_under_uniform_policy(state, 
                                               p_next_state_given_state_action: np.ndarray, 
                                               visualize=False,
                                               state_names=None):
    """ Calculate empowerment of state 'obs' by computing the mutual information under a uniform policy.
    
    Args:
        state: state for which to compute empowerment
        p_next_state_given_state_action: transition probabilities p(S_{t+1} | A_t^n, S_t=state), where A_t^n is a sequence of n action starting at time t. Matrix of shape (n_action_sequences, n_states)
        state_names: list of names of states to use in visualization function
        
    Returns:
        empowerment: empowerment value
    """
    
    # use uniform policy over action sequences
    p_action_given_state = np.ones(p_next_state_given_state_action.shape[0]) / p_next_state_given_state_action.shape[0] # p(A_t^n | S_t=s)
    
    # validate probability distributions
    validate_probability_distribution(p_action_given_state)
    validate_probability_distribution(p_next_state_given_state_action, axis=1)
    
    # Compute p(S_{t+n} | S_t=s) by marginalizing over A_t, A_{t+1}, ..., A_{t+n-1}
    p_next_state_given_state = p_next_state_given_state_action.T @ p_action_given_state # p(S_{t+n} | S_t=s)
    
    # Compute marginal entropy H(S_{t+n} | S_t=s)
    H_marginal = -np.sum(p_next_state_given_state * np.log2(p_next_state_given_state + 1e-8))
    
    # Compute conditional entropy H(S_{t+n} | A_t^n, S_t=s) = \sum_{a_t^n} p(a_t^n | s) H(S_{t+n} | a_t^n, s)
    H_conditional = 0
    for a in range(p_action_given_state.shape[0]):
        H_conditional += p_action_given_state[a] * -np.sum(p_next_state_given_state_action[a] * np.log2(p_next_state_given_state_action[a] + 1e-8))
    
    # I(A_t^n ; S_{t+n} | S_t=s) = H(S_{t+n} | S_t=s) - H(S_{t+n} | A_t^n, S_t=s)
    mutual_information = H_marginal - H_conditional
    
    if visualize:
        visualize_mutual_info_calculation(
            current_state=state, 
            p_next_state_given_state_action=p_next_state_given_state_action,
            p_action_given_state=p_action_given_state,
            state_names=state_names
        )
        plt.show()
    
    return mutual_information
    
def _rand_dist(shape):
    """ define a random probability distribution """
    P = np.random.rand(*shape)
    return _normalize(P)

def _normalize(P):
    """ normalize probability distribution """
    s = sum(P)
    if s == 0.:
        raise ValueError("input distribution has sum zero")
    return P / s
    
def blahut_arimoto_gopnik(P_yx: np.ndarray, state, epsilon=0.001, visualize_steps=False, state_names=None):
    """ 
    Compute the channel capacity C of a channel p(y|x) using the Blahut-Arimoto algorithm. To do
    this, finds the input distribution p(x) = p(A_t | S_t=s) that maximises the mutual information I(A_t; S_{t+1} | S_t=s)
    determined by p(y|x) = p(S_{t+1} | A_t, S_t=s) and p(x) = p(A_t | S_t). The channel capacity is then 
    C = max_{p(x)} I(x;y) = max_{p(A_t | S_t=s)} I(A_t; S_{t+1} | S_t=s).
    
    Code taken from the repo accompanying the following paper: Du, Y., Kosoy, E., Dayan, A., Rufova, M., Abbeel, P., & Gopnik, A. (2023, November). 
    What can AI Learn from Human Exploration? Intrinsically-Motivated Humans and Agents in Open-World Exploration. 
    In Neurips 2023 workshop: Information-theoretic principles in cognitive systems.
    
    (Original code source seems to be here though: https://github.com/Mchristos/empowerment/blob/master/empowerment/information_theory.py#L77)

    Args:
        P_yx: defines the channel p(y|x). Matrix of shape (n_y, n_x) where n_y is the number of output symbols and n_x is the number of input symbols.
        state: state for which to compute empowerment
        epsilon: convergence threshold
        visualize_steps: whether to visualize the terms in the mutual information calculation at each iteration
        state_names: list of names of states to use in visualization function
        
    Returns:
        C: channel capacity
    """
    eps = 1e-40
    P_yx = P_yx + eps # = p(S_t+1 | A_t, S_t=s)
    P_yx = P_yx / np.sum(P_yx, axis=0)
    
    # check that P_yx is a valid probability distribution 
    # (is of shape |S| x |A| and is a distribution over |S|, so should sum to 1 along axis 0)
    validate_probability_distribution(P_yx, axis=0)
        
    # initialize input dist randomly 
    q_x = _rand_dist((P_yx.shape[1],))
    T = 1
    
    iteration = 0

    while T > epsilon:
        # update PHI p(A_t | S_t+1, S_t) = p(S_t+1 | A_t, S_t) * p(A_t | S_t) / p(S_t+1 | S_t)
        PHI_yx = (P_yx*q_x.reshape(1,-1))/(P_yx @ q_x).reshape(-1,1)
        r_x = np.exp(np.sum(P_yx*np.log2(PHI_yx), axis=0))
        # channel capactiy 
        C = np.log2(np.sum(r_x))
        # check convergence 
        T = np.max(np.log2(r_x/q_x)) - C
        # update q
        q_x = _normalize(r_x + eps)
        
        if visualize_steps and (iteration % 10 == 0 or T <= epsilon): # visualize every 10 iterations or when converged
            if T <= epsilon:
                print(f"Converged after {iteration} iterations, T={T}, C={C}")
                print("Final source policy and next state distribution:")
            else:
                print(f"Iteration {iteration}, T={T}, C={C}")     
            
            visualize_mutual_info_calculation(
                current_state=state, 
                p_next_state_given_state_action=P_yx.T, # Transpose because P_yx is defined as |S| x |A|, but we want |A| x |S|
                p_action_given_state=q_x,
                state_names=state_names
            )
            plt.show()
        iteration += 1
    if C < 0:
        C = 0
    return C

def compute_empowerment_for_all_states(num_steps, method,
                                       env,
                                       num_samples=None,
                                       visualize_mutual_info_decomposition=False,
                                       states_to_visualize=[(0, 0)],
                                       transition_model: Optional[np.ndarray]=None,
                                       state_to_idx=None,
                                       action_to_idx=None,
                                       env_type="minigrid"):
    """ Compute empowerment for all states in the environment.

    Args:
        num_steps: Number of steps for empowerment calculation.
        method: Method to compute empowerment.
        num_samples: Number of samples for sampling-based methods.
        visualize_mutual_info_decomposition: Whether to visualize the mutual information decomposition.
        states_to_visualize: States to visualize the decomposition for.
        transition_model: Optional transition probabilities to use instead of env.transition_prob. Shape (|S|, |A|, |S|).
        state_to_idx: Mapping from state tuples to indices in the transition model.
        env_type: Type of environment (minigrid or gridworld).

    Returns:
        empowerment_map: 2D array of empowerment values for each state.
    """

    if env_type == "minigrid":
        # If using estimated transition model, adjust state and action spaces
        if transition_model is not None and state_to_idx is not None:
            validate_probability_distribution(transition_model, axis=2)
            idx_to_state = {v: k for k, v in state_to_idx.items()}
            num_states = len(state_to_idx)
        else:
            raise NotImplementedError

        empowerment_map = np.full((env.unwrapped.width, env.unwrapped.height, 4), np.nan)

        # Use the provided transition model
        T = transition_model

        # Generate all possible n-step action sequences using observed actions
        nstep_action_samples = list(itertools.product(list(range(T.shape[1])), repeat=num_steps))

        # Adjust T_n dimensions based on observed states and actions
        T_n = np.zeros([num_states, len(nstep_action_samples), num_states]) # P(S_{t+n} | A_t, A_{t+1}, ..., A_{t+n-1}, S_t)
        for i, an in enumerate(nstep_action_samples):
            T_n[:, i , :] = reduce((lambda x, y : np.dot(y, x)), map((lambda a : T[:,a,:]), an))
        
        for s_idx in range(num_states):
            s = idx_to_state[s_idx]
            x, y, dir = s
            if visualize_mutual_info_decomposition and (x, y) in states_to_visualize:
                visualize = True
            else:
                visualize = False
            if method == "count_unique_end_states_by_sampling":
                if num_samples is None:
                    raise ValueError("num_samples must be specified for method 'count_unique_end_states_by_sampling'")
                empowerment_map[x, y, dir] = calculate_empowerment_by_counting_unique_sampled_end_states(env, [x, y], num_steps, num_samples)
            elif method == "approximate_next_state_entropy_by_sampling":
                if num_samples is None:
                    raise ValueError("num_samples must be specified for method 'approximate_next_state_entropy_by_sampling'")
                empowerment_map[x, y, dir] = calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(env, [x, y], num_steps, num_samples)
            elif method == "blahut_arimoto":
                empowerment_map[x, y, dir] = blahut_arimoto_gopnik(T_n[s_idx,:,:].T, # Transpose because blahut_arimoto_gopnik expects P_yx to be |S| x |A|, but T_n[state,:,:] is |A| x |S|
                                                                state=s_idx, 
                                                                epsilon=1e-4,
                                                                visualize_steps=visualize,
                                                                state_names=[str(idx_to_state[i]) for i in range(num_states)])
            elif method == "marginalize_over_uniform_policy":
                empowerment_map[x, y, dir] = calculate_empowerment_under_uniform_policy(s_idx, T_n[s_idx,:,:], visualize=visualize, state_names=[str(idx_to_state[i]) for i in range(num_states)])
            else:
                raise ValueError(f"Unknown method: {method}")
    elif env_type == "gridworld":
        raise NotImplementedError("Gridworld empowerment calculation not implemented yet.")
        # If using estimated transition model, adjust state and action spaces
        if transition_model is not None and state_to_idx is not None:
            validate_probability_distribution(transition_model, axis=2)
            idx_to_state = {v: k for k, v in state_to_idx.items()}
            num_states = len(state_to_idx)
        else:
            idx_to_state = {env._get_state_index(x, y): (x, y) for x in range(env.width) for y in range(env.height)}
            num_states = env.num_states

        empowerment_map = np.full((env.width, env.height), np.nan)

        # Use the provided transition model or the environment's true probabilities
        T = transition_model if transition_model is not None else env.transition_prob

        # Generate all possible n-step action sequences using observed actions
        nstep_action_samples = list(itertools.product(list(range(T.shape[1])), repeat=num_steps))

        # Adjust T_n dimensions based on observed states and actions
        T_n = np.zeros([num_states, len(nstep_action_samples), num_states]) # P(S_{t+n} | A_t, A_{t+1}, ..., A_{t+n-1}, S_t)
        for i, an in enumerate(nstep_action_samples):
            T_n[:, i , :] = reduce((lambda x, y : np.dot(y, x)), map((lambda a : T[:,a,:]), an))
        
        for s_idx in range(num_states):
            s = idx_to_state[s_idx]
            x, y = s
            if (x, y) in env.obstacles:
                # Only compute empowerment for states the agent can actually be in
                empowerment_map[x, y] = np.nan
            else:
                if visualize_mutual_info_decomposition and (x, y) in states_to_visualize:
                    visualize = True
                else:
                    visualize = False
                if method == "count_unique_end_states_by_sampling":
                    if num_samples is None:
                        raise ValueError("num_samples must be specified for method 'count_unique_end_states_by_sampling'")
                    empowerment_map[x, y] = calculate_empowerment_by_counting_unique_sampled_end_states(env, [x, y], num_steps, num_samples)
                elif method == "approximate_next_state_entropy_by_sampling":
                    if num_samples is None:
                        raise ValueError("num_samples must be specified for method 'approximate_next_state_entropy_by_sampling'")
                    empowerment_map[x, y] = calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(env, [x, y], num_steps, num_samples)
                elif method == "blahut_arimoto":
                    empowerment_map[x, y] = blahut_arimoto_gopnik(T_n[s_idx,:,:].T, # Transpose because blahut_arimoto_gopnik expects P_yx to be |S| x |A|, but T_n[state,:,:] is |A| x |S|
                                                                    state=s_idx, 
                                                                    epsilon=1e-4,
                                                                    visualize_steps=visualize,
                                                                    state_names=[str(idx_to_state[i]) for i in range(num_states)])
                elif method == "marginalize_over_uniform_policy":
                    empowerment_map[x, y] = calculate_empowerment_under_uniform_policy(s_idx, T_n[s_idx,:,:], visualize=visualize, state_names=[str(idx_to_state[i]) for i in range(num_states)])
                else:
                    raise ValueError(f"Unknown method: {method}")
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented.")
    
    return empowerment_map


def compute_empowerment_for_state(state, num_steps, method,
                                  num_samples=None,
                                  visualize_mutual_info_decomposition=False,
                                  states_to_visualize=[(0, 0)],
                                  transition_model: Optional[np.ndarray]=None,
                                  state_to_idx=None,
                                  env_type="minigrid"):
    """ Compute empowerment for a single state in the environment.
    
    Args:
        state: State for which to compute empowerment as a tuple (x, y).
        num_steps: Number of steps for empowerment calculation.
        method: Method to compute empowerment.
        num_samples: Number of samples for sampling-based methods.
        visualize_mutual_info_decomposition: Whether to visualize the mutual information decomposition.
        states_to_visualize: States to visualize the decomposition for.
        transition_model: Optional transition probabilities to use instead of env.transition_prob. Shape (|S|, |A|, |S|).
        state_to_idx: Mapping from state tuples to indices in the transition model.
        env_type: Type of environment (minigrid or gridworld).
        
    Returns:
        empowerment: Empowerment value for the state.
    """

    if env_type == "minigrid":
        # If using estimated transition model, adjust state and action spaces
        if transition_model is not None and state_to_idx is not None:
            validate_probability_distribution(transition_model, axis=2)
            idx_to_state = {v: k for k, v in state_to_idx.items()}
            num_states = len(state_to_idx)
        else:
            raise NotImplementedError

        # Use the provided transition model
        T = transition_model

        # Generate all possible n-step action sequences using observed actions
        nstep_action_samples = list(itertools.product(list(range(T.shape[1])), repeat=num_steps))

        # Adjust T_n dimensions based on observed states and actions
        T_n = np.zeros([num_states, len(nstep_action_samples), num_states]) # P(S_{t+n} | A_t, A_{t+1}, ..., A_{t+n-1}, S_t)
        for i, an in enumerate(nstep_action_samples):
            T_n[:, i , :] = reduce((lambda x, y : np.dot(y, x)), map((lambda a : T[:,a,:]), an))
            
        empowerment = 0
        x, y, dir = state
        s_idx = state_to_idx[state]
        if visualize_mutual_info_decomposition and (x, y, dir) in states_to_visualize:
            visualize = True
        else:
            visualize = False
            
        if method == "count_unique_end_states_by_sampling":
            if num_samples is None:
                raise ValueError("num_samples must be specified for method 'count_unique_end_states_by_sampling'")
            empowerment = calculate_empowerment_by_counting_unique_sampled_end_states(env, [x, y], num_steps, num_samples)
        elif method == "approximate_next_state_entropy_by_sampling":
            if num_samples is None:
                raise ValueError("num_samples must be specified for method 'approximate_next_state_entropy_by_sampling'")
            empowerment = calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(env, [x, y], num_steps, num_samples)
        elif method == "blahut_arimoto":
            empowerment = blahut_arimoto_gopnik(T_n[s_idx,:,:].T, # Transpose because blahut_arimoto_gopnik expects P_yx to be |S| x |A|, but T_n[state,:,:] is |A| x |S|
                                                state=s_idx, 
                                                epsilon=1e-4,
                                                visualize_steps=visualize)
        elif method == "marginalize_over_uniform_policy":
            empowerment = calculate_empowerment_under_uniform_policy(s_idx, T_n[s_idx,:,:], visualize=visualize)
        else:
            raise ValueError(f"Unknown method: {method}")
    elif env_type == "gridworld":
        raise NotImplementedError("Gridworld empowerment calculation not implemented yet.")
        # If using estimated transition model, adjust state and action spaces
        if transition_model is not None and state_to_idx is not None:
            validate_probability_distribution(transition_model, axis=2)
            idx_to_state = {v: k for k, v in state_to_idx.items()}
            num_states = len(state_to_idx)
        else:
            idx_to_state = {env._get_state_index(x, y): (x, y) for x in range(env.width) for y in range(env.height)}
            num_states = env.num_states

        # Use the provided transition model or the environment's true probabilities
        T = transition_model if transition_model is not None else env.transition_prob

        # Generate all possible n-step action sequences using observed actions
        nstep_action_samples = list(itertools.product(list(range(T.shape[1])), repeat=num_steps))

        # Adjust T_n dimensions based on observed states and actions
        T_n = np.zeros([num_states, len(nstep_action_samples), num_states]) # P(S_{t+n} | A_t, A_{t+1}, ..., A_{t+n-1}, S_t)
        for i, an in enumerate(nstep_action_samples):
            T_n[:, i , :] = reduce((lambda x, y : np.dot(y, x)), map((lambda a : T[:,a,:]), an))
            
        empowerment = 0
        x, y = state
        s_idx = state_to_idx[state] if state_to_idx is not None else env._get_state_index(x, y)
        if (x, y) in env.obstacles:
            # throw error if trying to compute empowerment for an obstacle
            raise ValueError("Cannot compute empowerment for an obstacle state.")
        else:
            if visualize_mutual_info_decomposition and (x, y) in states_to_visualize:
                visualize = True
            else:
                visualize = False
                
            if method == "count_unique_end_states_by_sampling":
                if num_samples is None:
                    raise ValueError("num_samples must be specified for method 'count_unique_end_states_by_sampling'")
                empowerment = calculate_empowerment_by_counting_unique_sampled_end_states(env, [x, y], num_steps, num_samples)
            elif method == "approximate_next_state_entropy_by_sampling":
                if num_samples is None:
                    raise ValueError("num_samples must be specified for method 'approximate_next_state_entropy_by_sampling'")
                empowerment = calculate_empowerment_by_approximating_next_state_distribution_entropy_through_sampling(env, [x, y], num_steps, num_samples)
            elif method == "blahut_arimoto":
                empowerment = blahut_arimoto_gopnik(T_n[s_idx,:,:].T, # Transpose because blahut_arimoto_gopnik expects P_yx to be |S| x |A|, but T_n[state,:,:] is |A| x |S|
                                                    state=s_idx, 
                                                    epsilon=1e-4,
                                                    visualize_steps=visualize)
            elif method == "marginalize_over_uniform_policy":
                empowerment = calculate_empowerment_under_uniform_policy(s_idx, T_n[s_idx,:,:], visualize=visualize)
            else:
                raise ValueError(f"Unknown method: {method}")
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented.")
    return empowerment
        