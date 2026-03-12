"""
Count-based novelty for tabular environments.

Novelty of a state s is defined as the negative log of its empirical visit
probability:

    novelty(s) = -log p(s)  where  p(s) = n(s) / Σ_{s''} n(s'')

and n(s) is the total number of times state s has been visited (summed over all
actions and resulting next states in the transition counts).  High novelty means
the state has been visited rarely relative to the rest of the state space.

As exploration progresses and the visitation distribution flattens, novelty
converges toward a uniform (low) value everywhere.
"""

import numpy as np


def compute_novelty_for_state(state_idx: int, model) -> float:
    """Compute the novelty of a single state.

    Novelty is defined as -log p(s), where p(s) is the empirical fraction of
    transitions that originated in state s.

    Args:
        state_idx: Row index of the state in ``model.counts``
                   (i.e. ``model.state_to_idx[state]``).
        model: ``CountBasedTransitionModel`` instance.  Uses
               ``model.get_true_counts()`` to read visit counts.

    Returns:
        Novelty value in nats.  Higher means less visited.
    """
    raw_counts = model.get_true_counts()          # shape (|S|, |A|, |S|)
    state_visits = np.sum(raw_counts[state_idx])  # total transitions out of this state
    total_seen = np.sum(raw_counts)               # total transitions across all states

    p_s = state_visits / (total_seen + 1e-12)
    return -np.log(p_s + 1e-12)


def compute_novelty_for_all_states(model) -> np.ndarray:
    """Compute novelty for every state tracked by the model.

    Vectorised equivalent of calling :func:`compute_novelty_for_state` for each
    state index.

    Args:
        model: ``CountBasedTransitionModel`` instance.

    Returns:
        novelties: Array of novelty values of shape ``(n_states,)``, indexed by
                   ``model.state_to_idx``.  Higher means less visited.
    """
    raw_counts = model.get_true_counts()   # shape (|S|, |A|, |S|)

    # Sum outgoing transitions for each state (axis 1 = actions, axis 2 = next states)
    state_counts = np.sum(raw_counts, axis=(1, 2))   # shape (|S|,)
    total_seen = np.sum(state_counts)

    p_s = state_counts / (total_seen + 1e-12)
    novelties = -np.log(p_s + 1e-12)
    return novelties
