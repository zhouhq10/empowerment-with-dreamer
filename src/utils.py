from typing import Optional
import os
import numpy as np

from scipy.stats import entropy

def compute_transition_error(learned_matrix, learned_state_to_idx, true_matrix, true_state_to_idx):
    """Compute error between learned and true transition probabilities.
    
    Args:
        learned_matrix: numpy array of shape (n_observed_states, n_observed_actions, n_observed_states)
            containing learned transition probabilities
        learned_state_to_idx: dictionary mapping states to indices in learned_matrix
        true_matrix: numpy array of shape (n_total_states, n_actions, n_total_states)
            containing true transition probabilities
        true_state_to_idx: dictionary mapping states to indices in true_matrix
    
    Returns:
        error_matrix: numpy array of shape (n_total_states, ) containing
            the average error for each state (NaN for unobserved states)
    """
    
    n_total_states = true_matrix.shape[0]
    n_actions = true_matrix.shape[1]
    error_matrix = np.full((n_total_states,), np.nan)
    
    # Create inverse mapping from indices to states
    learned_idx_to_state = {v: k for k, v in learned_state_to_idx.items()}
    
    # For each known state
    for state, s_idx in learned_state_to_idx.items():
        true_s_idx = true_state_to_idx[state] # Get index for current state in true matrix
        error = 0
        
        # For each known action
        for action in range(n_actions):
            # Get learned distribution for this state-action pair
            learned_dist = learned_matrix[s_idx, action]
            
            # Get true distribution for this state-action pair
            true_dist = true_matrix[true_s_idx, action]
            
            # Convert learned distribution to use indices of ground truth matrix
            full_learned_dist = np.zeros(n_total_states)
            for learned_s_prime_idx, prob in enumerate(learned_dist):
                # Find the matching index of this next state in the true state space
                true_s_prime_idx = true_state_to_idx[learned_idx_to_state[learned_s_prime_idx]]
                full_learned_dist[true_s_prime_idx] = prob
            
            # Compute KL divergence between distributions
            eps = 1e-10 # Small constant to avoid division by zero
            error += entropy(true_dist + eps, full_learned_dist + eps) 
        
        # Average error over actions
        error_matrix[true_s_idx] = error / n_actions
            
    return error_matrix


def validate_probability_distribution(p: np.ndarray, axis: Optional[int] = None) -> None:
    """Validate probability distribution properties.
    
    Args:
        p: Array of probabilities
        axis: Axis along which distribution should sum to 1
        
    Raises:
        ValueError: If not a valid probability distribution
    """
    if not np.allclose(np.sum(p, axis=axis), 1.0, rtol=1e-5):
        # show those entries that are not close to 1
        invalid_entries = np.abs(np.sum(p, axis=axis) - 1.0) > 1e-5
        raise ValueError(f"Invalid probability distribution. Entries with indices {np.argwhere(invalid_entries)} do not sum to 1.")
    if not np.all(p >= 0):
        raise ValueError("Probabilities must be non-negative")