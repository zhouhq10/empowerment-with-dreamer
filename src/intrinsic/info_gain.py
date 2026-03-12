"""
Predicted Information Gain (PIG) for tabular environments.

Information gain measures how much an agent expects to learn about the transition
dynamics p(S' | S=s, A=a) by taking action a in state s.  It is defined as the
expected KL divergence between the posterior and the prior Dirichlet belief:

    PIG(s, a) = E_{s' ~ p(s'|s,a)} [ KL( p(S'|s,a,s') || p(S'|s,a) ) ]

Two estimation methods are provided:

* **LittleSommerPIG** — The closed-form expected forward-KL under a Dirichlet prior
  (Little & Sommer, 2011).  Used as the default in experiments.
* **dirichlet_entropy** — Entropy of the Dirichlet posterior; a cruder proxy for
  uncertainty that does not account for the direction of expected updates.

Reference:
    Little, D. Y., & Sommer, F. T. (2011). Learning in embodied action-perception
    loops through exploration. *arXiv preprint arXiv:1112.1125*.
"""

from typing import Literal

import numpy as np
import torch
from torch.distributions import Dirichlet

EPS = 1e-9

InfoGainMethod = Literal["LittleSommerPIG", "dirichlet_entropy"]


# ── Core information-theoretic primitives ──────────────────────────────────────

def calculate_kl_divergence_discrete(
    probs_prior: np.ndarray,
    probs_posterior: np.ndarray,
) -> float:
    """Compute KL(posterior ‖ prior) for two discrete probability distributions.

    Both inputs are normalised internally, so raw (unnormalised) count vectors are
    also accepted.  A small floor ``EPS`` is added before normalisation to avoid
    log(0).

    Args:
        probs_prior: Prior probability vector.  Shape ``(n,)``.
        probs_posterior: Posterior probability vector.  Shape ``(n,)``.

    Returns:
        KL divergence in bits, clamped to 0 to avoid negative values from
        floating-point rounding.  Returns ``np.nan`` on shape mismatch.
    """
    probs_prior = np.array(probs_prior) + EPS
    probs_posterior = np.array(probs_posterior) + EPS
    probs_prior /= np.sum(probs_prior)
    probs_posterior /= np.sum(probs_posterior)

    if probs_prior.shape != probs_posterior.shape:
        print(f"Error: Shape mismatch for KL Discrete. Prior: {probs_prior.shape}, Post: {probs_posterior.shape}")
        return np.nan

    kl_div = np.sum(probs_posterior * np.log2(probs_posterior / probs_prior))
    return max(kl_div, 0.0)


def calculate_expected_discrete_forward_KL(
    counts: np.ndarray,
    update_strength: float,
) -> float:
    """Compute the Predicted Information Gain (PIG) for one state-action pair.

    Implements the Little & Sommer (2011) estimator.  For a Dirichlet belief
    Dir(α) over next states, PIG is:

        PIG = Σ_{s'} p(s'|s,a) · KL( p(S'|s,a,s') ‖ p(S'|s,a) )

    where p(s'|s,a) = α_s' / Σ α and the posterior α' is obtained by adding
    ``update_strength`` to the count for s'.

    Args:
        counts: Dirichlet concentration parameters α for all next states.
                Shape ``(n_states,)``.  Typically ``model.counts[s_idx, action]``.
        update_strength: How much a single observed transition increments a count.
                         Matches ``CountBasedTransitionModel.update_strength``.

    Returns:
        Expected KL divergence (PIG) in bits.
    """
    # Add EPS for numerical stability before computing the prior mean
    alphas = np.array(counts, dtype=float) + EPS
    alpha_0 = np.sum(alphas)

    prior_probs = alphas / alpha_0  # current mean estimate p(S'|s,a)

    # For each possible next state s', compute how the belief would shift
    # if that transition were observed: KL( posterior || prior )
    kl_divs = np.zeros_like(alphas)
    for i in range(len(alphas)):
        posterior_alphas = alphas.copy()
        posterior_alphas[i] += 1.0 * update_strength
        posterior_probs = posterior_alphas / np.sum(posterior_alphas)
        kl_divs[i] = calculate_kl_divergence_discrete(posterior_probs, prior_probs)

    # Weight each KL by the probability that next state s' actually occurs
    # PIG = Σ_{s'} p(s'|s,a) · KL(...)
    return np.sum(prior_probs * kl_divs)


# ── Per-state and per-state-action estimators ──────────────────────────────────

def calculate_predicted_information_gain_for_state_action_pair(
    state,
    action: int,
    model,
    method: InfoGainMethod = "LittleSommerPIG",
) -> float:
    """Compute predicted information gain for a single (state, action) pair.

    Args:
        state: State identifier (any hashable type accepted by ``model.state_to_idx``).
        action: Action index.
        model: ``CountBasedTransitionModel`` instance providing ``counts``,
               ``update_strength``, and ``observed_states``.
        method: Estimation method — ``"LittleSommerPIG"`` (default) or
                ``"dirichlet_entropy"``.

    Returns:
        Predicted information gain in bits (or nats for ``dirichlet_entropy``).
        Returns 0.0 if the model has no observed states yet.
    """
    if not model.observed_states:
        return 0.0

    s_idx = model.state_to_idx[state]
    counts = model.counts[s_idx, action]

    if method == "dirichlet_entropy":
        # Entropy of the Dirichlet posterior — a proxy for overall uncertainty
        alpha = torch.tensor(counts, dtype=torch.float32)
        dist = Dirichlet(alpha)
        return dist.entropy().item()

    elif method == "LittleSommerPIG":
        return calculate_expected_discrete_forward_KL(
            counts, update_strength=model.update_strength
        )

    else:
        raise NotImplementedError(f"Method {method} not implemented.")


def calculate_predicted_information_gain_for_state(
    state,
    model,
    method: InfoGainMethod = "LittleSommerPIG",
) -> float:
    """Compute the action-averaged predicted information gain for a state.

    Averages ``calculate_predicted_information_gain_for_state_action_pair`` over
    all actions uniformly:

        PIG(s) = (1 / |A|) · Σ_a PIG(s, a)

    Args:
        state: State identifier.
        model: ``CountBasedTransitionModel`` instance.
        method: Estimation method (forwarded to the per-pair function).

    Returns:
        Mean predicted information gain across actions, in bits.
    """
    info_gain = 0.0
    for action in range(model.num_actions):
        info_gain += calculate_predicted_information_gain_for_state_action_pair(
            state, action, model, method=method
        )
    # Average over actions
    info_gain /= model.num_actions
    return info_gain


# ── Batch helper ───────────────────────────────────────────────────────────────

def compute_information_gain_for_all_states(
    env,
    model,
    method: InfoGainMethod = "LittleSommerPIG",
    env_type: Literal["minigrid", "gridworld"] = "minigrid",
) -> np.ndarray:
    """Compute action-averaged PIG for every state in the environment.

    Iterates over ``model.observed_states`` and calls
    :func:`calculate_predicted_information_gain_for_state` for each.

    Args:
        env: Environment instance.  Used only to read grid dimensions and
             ``env.obstacles`` (GridWorld) or ``env.unwrapped.width/height``
             (MiniGrid).
        model: ``CountBasedTransitionModel`` instance with populated counts.
        method: Estimation method.
        env_type: ``"minigrid"`` or ``"gridworld"``.

    Returns:
        info_gain_map: Array of PIG values indexed by state coordinates.
            Shape ``(width, height, 4)`` for MiniGrid (last axis = orientation) or
            ``(width, height)`` for GridWorld.  Obstacle states are NaN.

    Note:
        The map stores state-averaged (not action-resolved) values.  For
        per-action resolution, call
        :func:`calculate_predicted_information_gain_for_state_action_pair` directly.
    """
    if env_type == "minigrid":
        info_gain_map = np.full((env.unwrapped.width, env.unwrapped.height, 4), np.nan)

        for state in model.observed_states:
            x, y, dir = state
            info_gain_map[x, y, dir] = calculate_predicted_information_gain_for_state(
                state, model, method=method
            )

    elif env_type == "gridworld":
        info_gain_map = np.full((env.width, env.height), np.nan)

        for state in model.observed_states:
            x, y = state
            if (x, y) in env.obstacles:
                # Agent can never occupy obstacle cells — leave as NaN
                info_gain_map[x, y] = np.nan
            else:
                info_gain_map[x, y] = calculate_predicted_information_gain_for_state(
                    state, model, method=method
                )

    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented.")

    return info_gain_map
