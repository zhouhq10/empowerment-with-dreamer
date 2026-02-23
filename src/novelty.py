import numpy as np

def compute_novelty_for_state(state_idx, model):
    """Compute novelty of a state based on transition model.
    
    Args:
        state_idx: State for which to compute novelty
        model: CountBasedTransitionModel object
    
    Returns:
        novelty: Novelty of the state
    """
    raw_counts = model.get_true_counts()
    state_visits = np.sum(raw_counts[state_idx])
    total_seen = np.sum(raw_counts)

    p_s = state_visits / (total_seen + 1e-12)
    return -np.log(p_s + 1e-12)

def compute_novelty_for_all_states(model):
    """Compute novelty for all states in the model.
    
    Args:
        model: CountBasedTransitionModel object
    
    Returns:
        novelties: Array of novelty values for each state
    """
    raw_counts = model.get_true_counts() # (|S|, |A|, |S|)
    
    # Compute novelty for each state by summing over last two dimensions
    state_counts = np.sum(raw_counts, axis=(1, 2)) 
    total_seen = np.sum(state_counts)
    p_s = state_counts / (total_seen + 1e-12)
    novelties = -np.log(p_s + 1e-12)
    return novelties

