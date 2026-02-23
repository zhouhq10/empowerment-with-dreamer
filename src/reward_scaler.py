import numpy as np

class SlidingWindowRewardScaler:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reward_history = []
        
    def update(self, reward):
        self.reward_history.append(reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            
    def scale(self, reward):
        if not self.reward_history:
            return 0
            
        reward_min = min(self.reward_history)
        reward_max = max(self.reward_history)
        if reward_min != reward_max:
            scaled_reward = (reward - reward_min) / (reward_max - reward_min)
        else:
            scaled_reward = 0
            
        return scaled_reward
    
class GlobalRewardScaler:
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def update(self, reward):
        """
        Updates the global min and max values based on the input reward(s).

        Args:
            reward: A scalar reward or a NumPy array of rewards.
        """
        if isinstance(reward, np.ndarray):
            if reward.size > 0: # Ensure array is not empty
                self.min_val = min(self.min_val, np.min(reward))
                self.max_val = max(self.max_val, np.max(reward))
        else:
            # Handle scalar input
            self.min_val = min(self.min_val, reward)
            self.max_val = max(self.max_val, reward)
            
    def scale(self, reward):
        """
        Scales the input reward(s) to the range [0, 1] based on the
        globally observed min and max values.

        Args:
            reward: A scalar reward or a NumPy array of rewards.

        Returns:
            A scalar or a NumPy array with the scaled reward(s).
            Returns 0 or an array of zeros if min == max.
        """
        if self.min_val == self.max_val:
            # Handle scalar or array input for the zero case
            if isinstance(reward, np.ndarray):
                return np.zeros_like(reward)
            else:
                return 0.0

            
        scaled_reward = (reward - self.min_val) / (self.max_val - self.min_val)
        return scaled_reward