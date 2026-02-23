import numpy as np
import torch
from torch.distributions import Dirichlet

EPS = 1e-9 
    
def calculate_kl_divergence_discrete(probs_prior, probs_posterior):
    """Calculates KL(posterior || prior) for discrete probability distributions."""
    probs_prior = np.array(probs_prior) + EPS
    probs_posterior = np.array(probs_posterior) + EPS
    probs_prior /= np.sum(probs_prior)
    probs_posterior /= np.sum(probs_posterior)

    if probs_prior.shape != probs_posterior.shape:
        print(f"Error: Shape mismatch for KL Discrete. Prior: {probs_prior.shape}, Post: {probs_posterior.shape}")
        return np.nan

    kl_div = np.sum(probs_posterior * np.log2(probs_posterior / probs_prior))
    return max(kl_div, 0.0)

def calculate_expected_discrete_forward_KL(counts, update_strength):
    # As in: Little, D. Y., & Sommer, F. T. (2011). Learning in embodied action-perception loops through exploration.
    # Also known as PIG (predicted information gain) in their work

    # Ensure counts are float and add EPS for numerical stability
    alphas = np.array(counts, dtype=float) + EPS
    alpha_0 = np.sum(alphas)

    # compute kl divergence per state:
    prior_probs = alphas / alpha_0
    kl_divs = np.zeros_like(alphas)
    for i in range(len(alphas)):
      # Compute how probabilities would change if one of the next states S'=s' would be observed
      # The change in distribution can be measured by KL(p(S'|s,a,s') || p(S'|s,a))
      posterior_alphas = alphas.copy()
      posterior_alphas[i] += 1.0 * update_strength
      posterior_probs = posterior_alphas / np.sum(posterior_alphas)
      kl_divs[i] = calculate_kl_divergence_discrete(posterior_probs, prior_probs)

    # To compute the *expected* KL divergence, we have to weigh each possible change by the probability that it's cause (a certain s') occurs
    # And the probability with which we expect it to occur is exactly p(s'|s,a)
    # PIG = sum_{s'} [ p(s'|s,a) * KL(p(S'|s,a,s') || p(S'|s,a)) ]
    return np.sum(prior_probs * kl_divs)

# similar function to compute_empowerment_for_all_states, but instead computing average predicted info gain for all states
def compute_information_gain_for_all_states(env, model, method='LittleSommerPIG', env_type='minigrid'):
    """ Compute average predicted information gain for all states in the environment.

    Args:
        env: GridWorld environment.
        model: CountBasedTransitionModel object, whose counts will define a Dirichlet distribution for each state-action pair
        env_type: Type of environment (minigrid or gridworld).

    Returns:
        info_gain_map: 2D array of information gain values for each state.
    """

    # TODO: Maybe better save for each state-action pair rather than each state only for more detailed analysis

    if env_type == 'minigrid':
        info_gain_map = np.full((env.unwrapped.width, env.unwrapped.height, 4), np.nan)

        for state in model.observed_states:
            x, y, dir = state
            info_gain_map[x, y, dir] = calculate_predicted_information_gain_for_state(state, model, method=method)

    elif env_type == 'gridworld':
        info_gain_map = np.full((env.width, env.height), np.nan)

        for state in model.observed_states:
            x, y = state
            if (x, y) in env.obstacles:
                # Only compute info gain for states the agent can actually be in
                info_gain_map[x, y] = np.nan
            else:
                info_gain_map[x, y] = calculate_predicted_information_gain_for_state(state, model, method=method)
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented.")

    return info_gain_map

def calculate_predicted_information_gain_for_state(state, model, method='LittleSommerPIG'):
    """ Compute predicted information gain for a single state s by 

    Args:
        state: State for which to compute information gain.
        model: CountBasedTransitionModel object.

    Returns:
        info_gain: Information gain value for the state.
    """
    info_gain = 0.0

    for action in range(model.num_actions):
        info_gain += calculate_predicted_information_gain_for_state_action_pair(state, action, model, method=method)

    # average over actions
    info_gain /= model.num_actions

    return info_gain

def calculate_predicted_information_gain_for_state_action_pair(state, action, model, method='LittleSommerPIG'):
    """ Compute predicted information gain for a state-action-pair.

    Args:
        state: State for which to compute information gain.
        model: CountBasedTransitionModel object.

    Returns:
        info_gain: Information gain value for the state.
    """
    
    if not model.observed_states:
        return 0.0
    
    if method == 'dirichlet_entropy':
        # Get counts numpy array for all possible next states
        s_idx = model.state_to_idx[state]
        counts = model.counts[s_idx, action]

        # Convert to tensor
        alpha = torch.tensor(counts, dtype=torch.float32)

        # Create Dirichlet distribution
        dist = Dirichlet(alpha)

        # Add to total information gain
        return dist.entropy().item()
    elif method == "LittleSommerPIG":
        update_strength = model.update_strength
        # Get counts numpy array for all possible next states
        s_idx = model.state_to_idx[state]
        counts = model.counts[s_idx, action]

        # Compute expected KL divergence
        return calculate_expected_discrete_forward_KL(counts, update_strength=update_strength)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")
    
    