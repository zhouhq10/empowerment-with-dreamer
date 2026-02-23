from collections import defaultdict
import numpy as np

from info_gain import calculate_predicted_information_gain_for_state_action_pair, compute_information_gain_for_all_states
from empowerment import compute_empowerment_for_all_states, compute_empowerment_for_state
from novelty import compute_novelty_for_state, compute_novelty_for_all_states
from reward_scaler import GlobalRewardScaler

class CountBasedTransitionModel:
    def __init__(self, 
                 num_actions: int,
                 states: list, 
                 prior_count=1, 
                 update_strength=None, 
                 reward_types=[], 
                 reward_configs: dict[str, dict] | None = None, 
                 combination_method="mean", 
                 is_true_model=False):
        """Transition model based on counting observed transitions.
        
        Args:
            num_actions: Number of actions in the environment
            prior_count: Small initial value for counts to avoid zero probabilities (essentially laplace smoothing)
            update_strength: How much to update counts for each observed transition
            states: List of all possible states in the environment
            reward_types: List of reward types to be used in the model. Can be any of ["extrinsic", "novelty", "info_gain", "empowerment"]
            reward_configs is a dict that maps a reward name to the kwargs that should be forwarded to the corresponding reward function.
            combination_method: How to combine multiple reward types if more than one is used. Can be "mean" or "product".
            is_true_model: If True, the model is not updated and considered final.
        """
        
        # All states
        self.observed_states = states
        self.state_to_idx = {s: i for i, s in enumerate(states)}

        self.prior_count = prior_count
        self.update_strength = 100*len(states) if update_strength is None else update_strength
        self.num_actions = num_actions
        
        # Initialize numpy array of counts for each transition with prior counts
        self.counts = np.ones((len(states), num_actions, len(states))) * prior_count
        # Initialize numpy array of terminal transitions
        # This is a boolean array indicating whether the transition is terminal
        self.terminal_transitions = np.zeros((len(states), num_actions, len(states)), dtype=bool)
        # Initialize total (intrinsic) reward
        self.R = np.zeros((len(states), num_actions, len(states)))

        self.reward_types = reward_types
        self.combination_method = combination_method
        self.reward_configs = reward_configs if reward_configs is not None else {}
            
        # Initialize intrinsic rewards
        self._initialize_intrinsic_rewards()

        self.is_true_model = is_true_model
        
        
    def update(self, state, action, next_state, extrinsic_reward, terminated):
        """Update counts based on observed transition."""
        s = self.state_to_idx[state]
        a = action
        s_prime = self.state_to_idx[next_state]

        if not self.is_true_model:
            self.counts[s, a, s_prime] += self.update_strength
            if terminated:
                self.terminal_transitions[s, a, s_prime] = True

            # Update intrinsic rewards
            self._update_intrinsic_rewards(state, action, next_state)

         # TODO: Implement extrinsic reward handling

    def _update_intrinsic_rewards(self, state, action, next_state):
        """Update intrinsic rewards based on observed transition."""

        s = self.state_to_idx[state]
        a = action
        s_prime = self.state_to_idx[next_state]

        if "novelty" in self.reward_types:
            cfg = self.reward_configs.get("novelty", {}) 
            if cfg:
                raise NotImplementedError("Novelty reward does not take any config at the moment, are you sure you wanted to pass this config?")
            
            # novelty cannot be cached, since it changes with every update 
            # (how novel one state is depends on how often it has been visited in relation to others)
            # thus, we need to recompute it for all states
            novelty_per_state = compute_novelty_for_all_states(self) # array of shape (|S|)
            
            # Assign novelty based on the next state s'. Broadcast the (|S|,) array to (|S|, |A|, |S|)
            self.raw_reward_components['novelty'][:, :, :] = novelty_per_state

            # rescale entire reward component to [0, 1]
            self.reward_scalers["novelty"].update(novelty_per_state)
            self.scaled_reward_components['novelty'] = self.reward_scalers["novelty"].scale(
                self.raw_reward_components['novelty']
                )

        if "empowerment" in self.reward_types:
            cfg = self.reward_configs.get("empowerment", {})
            learned_matrix, learned_state_to_idx = self.get_full_transition_matrix()
            
            num_steps_emp = cfg["num_steps"]
            if num_steps_emp != 1:
                raise NotImplementedError(f"Empowerment for {num_steps_emp} steps requires changed caching behavior, which is not implemented yet.") 

            # to save computation, only update empowerment of those state whose model could have changed, i.e. the current one
            changed_empowerment = compute_empowerment_for_state(
                state, 
                transition_model=learned_matrix, 
                state_to_idx=learned_state_to_idx,
                **cfg
                )
            
            # cache value
            self.raw_reward_components['empowerment'][:, :, s] = changed_empowerment

            # rescale entire reward component to [0, 1]
            self.reward_scalers["empowerment"].update(changed_empowerment)
            self.scaled_reward_components['empowerment'] = self.reward_scalers["empowerment"].scale(
                self.raw_reward_components['empowerment']
                )
            
        if "info_gain" in self.reward_types:
            cfg = self.reward_configs.get("info_gain", {})
            # to save computation, only update info gain of those state whose model could have changed, i.e. the current one
            changed_info_gain = calculate_predicted_information_gain_for_state_action_pair(
                state, action, self, **cfg
                )
            
            # cache value
            self.raw_reward_components['info_gain'][s, a] = changed_info_gain

            # rescale entire reward component to [0, 1]
            self.reward_scalers["info_gain"].update(changed_info_gain)
            self.scaled_reward_components['info_gain'] = self.reward_scalers["info_gain"].scale(
                self.raw_reward_components['info_gain']
                )            

        # Combine rewards into single R matrix
        if len(self.reward_types) == 0:
            # TODO: maybe compare "constant 1" here
            pass
        elif len(self.reward_types) == 1:
            self.R[:] = self.scaled_reward_components[self.reward_types[0]]
        elif len(self.reward_types) > 1:
            if self.combination_method == "mean":
                self.R[:] = np.mean([self.scaled_reward_components[r] for r in self.reward_types], axis=0)
            elif self.combination_method == "product":
                self.R[:] = np.prod([self.scaled_reward_components[r] for r in self.reward_types], axis=0)

    def _initialize_intrinsic_rewards(self):
        # in the first step, the agent has no reward model yet, so we have to initialize it with the rewards under the current model
        print("Updating rewards in first step")
            
        # Initialize reward scalers, raw components and scaled components
        self.reward_scalers = {}
        self.raw_reward_components = {}
        self.scaled_reward_components = {}
        for reward_type in self.reward_types:
            if reward_type in ["novelty", "info_gain", "empowerment"]:
                self.reward_scalers[reward_type] = GlobalRewardScaler()
                self.raw_reward_components[reward_type] = np.zeros_like(self.R)
                self.scaled_reward_components[reward_type] = np.zeros_like(self.R)
            else:
                raise ValueError(f"Unknown reward type: {reward_type}")

        if "empowerment" in self.reward_types:
            cfg = self.reward_configs.get("empowerment", {})
            learned_matrix, learned_state_to_idx = self.get_full_transition_matrix()

            # First, compute reward for all states before scaling
            for state in self.observed_states:
                state_idx = self.state_to_idx[state]
                empowerment_reward = compute_empowerment_for_state(
                    state, 
                    transition_model=learned_matrix, 
                    state_to_idx=learned_state_to_idx,
                    **cfg
                )
                self.reward_scalers["empowerment"].update(empowerment_reward)
                self.raw_reward_components["empowerment"][:, :, state_idx] = empowerment_reward

            # Then, scale reward
            self.scaled_reward_components["empowerment"] = self.reward_scalers["empowerment"].scale(
                self.raw_reward_components["empowerment"]
                )

        if "info_gain" in self.reward_types:
            cfg = self.reward_configs.get("info_gain", {})
            # First, compute reward for all states before scaling
            for state in self.observed_states:
                state_idx = self.state_to_idx[state]
                for action in range(self.num_actions):
                    info_gain_reward = calculate_predicted_information_gain_for_state_action_pair(
                        state, action, self, **cfg
                        )
                    self.reward_scalers["info_gain"].update(info_gain_reward)
                    self.raw_reward_components["info_gain"][state_idx, action] = info_gain_reward

            # Then, scale reward
            self.scaled_reward_components["info_gain"] = self.reward_scalers["info_gain"].scale(
                self.raw_reward_components["info_gain"]
                )

        if "novelty" in self.reward_types:
            cfg = self.reward_configs.get("novelty", {})
            # First, compute reward for all states before scaling
            for state in self.observed_states:
                state_idx = self.state_to_idx[state]
                novelty_raw = compute_novelty_for_state(state_idx, self, **cfg)
                self.reward_scalers["novelty"].update(novelty_raw)
                self.raw_reward_components["novelty"][state_idx, :, :] = novelty_raw

            # Then, scale reward
            self.scaled_reward_components["novelty"] = self.reward_scalers["novelty"].scale(
                self.raw_reward_components["novelty"]
                )

        # Combine rewards into single R matrix
        if len(self.reward_types) == 0:
            # TODO: maybe compare "constant 1" here
            pass
        elif len(self.reward_types) == 1:
            self.R[:] = self.scaled_reward_components[self.reward_types[0]]
        elif len(self.reward_types) > 1:
            if self.combination_method == "mean":
                self.R[:] = np.mean([self.scaled_reward_components[r] for r in self.reward_types], axis=0)
            elif self.combination_method == "product":
                self.R[:] = np.prod([self.scaled_reward_components[r] for r in self.reward_types], axis=0)


        print("Reward components initialized in first step")
    
    def get_full_transition_matrix(self, dirichlet=False, random_state=42):
        """Get transition probability matrix p(s'|s,a) for observed state-action pairs.

        Args:
            dirichlet: If True, sample from the Dirichlet instead of returning the mean.
        
        Returns:
            matrix: numpy array of shape (num_observed_states, num_observed_actions, num_observed_states)
            state_to_idx: dictionary mapping states to matrix indices
        """
        
        if not dirichlet:
            # Return the mean of the dirichlet
            matrix = self.counts / np.sum(self.counts, axis=2, keepdims=True)

            # Round the transition probabilities to reduce precision of extremely small values
            matrix = np.round(matrix, decimals=2)

            # Ensure proper normalization after rounding
            # To handle potential rounding errors that make rows not sum to 1, renormalize.
            # set those entries that are all zero to 1 instead
            matrix[np.sum(matrix, axis=2) == 0] = 1
            matrix = matrix / np.sum(matrix, axis=2, keepdims=True)
        else:
            # Sample from the dirichlet 
            rng = np.random.default_rng(random_state)
            # sampling from gamma as below is equivalent to sampling from dirichlet
            # and allows vectorized implementation
            G = rng.gamma(shape=self.counts, scale=1.0) 
            matrix = G / G.sum(axis=2, keepdims=True)


        return matrix, self.state_to_idx
    
    def get_true_counts(self):
        """
        Return a 3D array of shape [n_states, n_actions, n_states], 
        where each entry is the actual number of observed transitions,
        i.e., (counts[s,a,s'] - prior_count) / update_strength.
        """
        raw_counts = (self.counts - self.prior_count) / self.update_strength
        return raw_counts
