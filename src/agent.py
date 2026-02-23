from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np
from scipy.special import softmax

from transition_model import CountBasedTransitionModel

class Agent(ABC):
    """Abstract base class for all agents."""
    
    @abstractmethod
    def select_action(self, state):
        """Select an action given the current state.
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action
        """
        pass
    
    @abstractmethod
    def update(self, state, action, next_state, reward, terminated):
        """Update agent's internal state after a transition.
        
        Args:
            state: Current state
            action: Taken action
            next_state: Resulting state
            reward: Received reward
            terminated: Whether the episode has terminated
        """
        pass

class RandomAgent(Agent):
    """Agent that selects actions randomly."""
    
    def __init__(self, num_actions):
        self.num_actions = num_actions
        
    def select_action(self, state):
        return np.random.choice(self.num_actions)
        
    def update(self, state, action, next_state, reward, terminated):
        # Random agent doesn't need to learn anything
        pass

class PrioritizedSweepingAgent(Agent):
    """ Model-based agent that stores Q-values and updates them using prioritized sweeping. 
        Adapted from: https://github.com/gruaz-lucas/Merits_of_curiosity/blob/main/src/agent.py
    """
    def __init__(
        self,
        num_actions,
        all_states,
        gamma=0.9,
        learning_rate=1.0,
        temperature=0.1,
        n_sweeps=None,
        q_init=0.0,
        random_state=42,
        model_kwargs: dict | None = None,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.lr = learning_rate
        self.temperature = temperature
        self.q_init = q_init
        self.T_PS = n_sweeps if n_sweeps is not None else 100*len(all_states)

        # State management
        self.state_to_idx = {s: i for i, s in enumerate(all_states)}
        self.n_states = len(all_states)
        
        # Initialize value functions
        self.q_table = np.full((self.n_states, num_actions), q_init)
        self.U = np.max(self.q_table, axis=1)
        
        # Initialize the transition model
        self.model = CountBasedTransitionModel(num_actions, states=all_states, **(model_kwargs or {}))
        
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def select_action(self, state):
        state_idx = self.state_to_idx[state]
        q_values = self.q_table[state_idx]
        # Normalize q-values to [0,1] to ensure same behavior regardless of gamma
        # MAX_VALUE = 1.0 / (1.0 - self.gamma) # Theoretical maximum value given rewards in [0,1]
        # π = softmax(q_values / MAX_VALUE)
        # π = softmax(q_values)

        # use temperature scaling
        # π = softmax(q_values / self.temperature)

        # if np.any(np.isnan(π)):
        #     # if exp overflows, set to 1.0 and normalize
        #     π = np.nan_to_num(π, nan=1.0)
        #     π /= np.sum(π)
        # return self.rng.choice(self.num_actions, p=π)

        # Changed to greedy action selection because intrinsic rewards should drive exploration, not arbitrary randomness
        # Greedy action selection:
        # (but make sure to select a random best action if multiple Q-values are the same)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        action = self.rng.choice(best_actions)  
        return action
        

    def update(self, state, action, next_state, reward, terminated):
        # Update the model
        self.model.update(state, action, next_state, reward, terminated)

        # Perform prioritized sweeping
        self.prioritized_sweeping()

    def prioritized_sweeping(self, ΔV_thresh=1e-8, theta_thresh=1e-8):
        λ = self.gamma
        Q = self.q_table
        U = self.U
        R = self.model.R
        T, _ = self.model.get_full_transition_matrix()
        D = self.model.terminal_transitions # Transitions that lead to episode termination (will zero out future rewards)
        
        u_reshaped = np.tile(U, (self.n_states, self.num_actions, 1))
        R_scaled = R * (1 - λ) # Scale R so that Q-values are in [0,1] later
        r_plus_u = R_scaled + λ * u_reshaped * (~D)
        
        Q[:] = np.sum(T * r_plus_u, axis=2)

        V = np.max(Q, axis=1)
        priorities = np.abs(U - V)
        
        for _ in range(self.T_PS):
            s_prime = np.argmax(priorities)
            ΔV = V[s_prime] - U[s_prime]
            
            if np.max(V) != np.min(V) and abs(ΔV) / abs(np.max(V) - np.min(V)) <= ΔV_thresh:
                #print(abs(ΔV))
                #print(f"Converged within threshold: {abs(ΔV) / abs(np.max(V) - np.min(V))} <= {ΔV_thresh}")
                break

            U[s_prime] = V[s_prime]

            # Set priority to 0 after processing to prevent immediate re-selection
            priorities[s_prime] = 0.0 
            
            mask = T[:, :, s_prime] * np.abs(ΔV) > theta_thresh
            states_to_update = np.where(mask.any(axis=1))[0]

            # print(mask.shape)
            # print(f"States where one action is likely to lead to {mask[mask.any(axis=1)]} > {theta_thresh}")
            # print(f"Number of states to update: {len(states_to_update)}")
            # print(f"States to update: {states_to_update}")

            if states_to_update.size > 0:
                Q[states_to_update, :] = Q[states_to_update, :] + λ * T[states_to_update, :, s_prime] * ΔV
                V[states_to_update] = np.max(Q[states_to_update, :], axis=1)
                priorities[states_to_update] = np.abs(U[states_to_update] - V[states_to_update])

        if _ == self.T_PS - 1:
            pass # Skipping temporarily because the print statements clutter the output in the notebook, should probably log to logger instead
            # print("Warning: Prioritized sweeping did not converge within the specified number of iterations.")

        self.q_table = Q
        self.U = U

        return
    
    def reset_model(self):
        raise NotImplementedError("Resetting the model is not implemented for this agent.")
        
        # Reset the value functions
        self.q_table.fill(self.q_init)
        self.U = np.max(self.q_table, axis=1)