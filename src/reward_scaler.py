"""Reward scaling utilities for normalizing intrinsic reward signals.

Two strategies are provided:
- ``SlidingWindowRewardScaler``: normalizes to [0, 1] relative to the most
  recent *window_size* reward values (adapts to recent distribution shifts).
- ``GlobalRewardScaler``: normalizes to [0, 1] relative to the global min/max
  seen since the scaler was created (stable, monotone bounds).

Both scalers return 0 (or an all-zero array) when min == max to avoid
division-by-zero.
"""

from __future__ import annotations

import numpy as np


class SlidingWindowRewardScaler:
    """Min-max scaler that tracks a fixed-size window of recent reward values.

    Args:
        window_size: Number of past values to retain for normalization.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.reward_history: list[float] = []

    def update(self, reward: float) -> None:
        """Append *reward* to the history, evicting the oldest entry if full.

        Args:
            reward: Scalar reward value to record.
        """
        self.reward_history.append(reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)

    def scale(self, reward: float | np.ndarray) -> float | np.ndarray:
        """Scale *reward* to [0, 1] using the current window's min/max.

        Args:
            reward: Scalar or array of reward values to normalize.

        Returns:
            Normalized reward(s); 0 when the window is empty or min == max.
        """
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
    """Min-max scaler that tracks the global min/max over all observed rewards.

    Unlike ``SlidingWindowRewardScaler``, the bounds only ever widen, making
    this suitable when the reward range is expected to grow monotonically.
    """

    def __init__(self) -> None:
        self.min_val: float = float('inf')
        self.max_val: float = float('-inf')

    def update(self, reward: float | np.ndarray) -> None:
        """Expand the tracked range to include *reward*.

        Args:
            reward: A scalar reward or a NumPy array of rewards.
        """
        if isinstance(reward, np.ndarray):
            if reward.size > 0:  # Ensure array is not empty
                self.min_val = min(self.min_val, np.min(reward))
                self.max_val = max(self.max_val, np.max(reward))
        else:
            # Handle scalar input
            self.min_val = min(self.min_val, reward)
            self.max_val = max(self.max_val, reward)

    def scale(self, reward: float | np.ndarray) -> float | np.ndarray:
        """Scale *reward* to [0, 1] using the globally observed min/max.

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
