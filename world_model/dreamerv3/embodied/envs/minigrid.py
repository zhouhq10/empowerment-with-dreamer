# Code obtained from qxcv: https://gist.github.com/qxcv/e8641342c102c2aa714c9caeca724101

import numpy as np
from typing import cast, Any, TypeVar
import gymnasium
from gymnasium import spaces
from minigrid.wrappers import RGBImgObsWrapper, ObservationWrapper, ImgObsWrapper, RGBImgPartialObsWrapper, FullyObsWrapper
from dreamerv3.embodied.core.wrappers import ResizeImage
from dreamerv3.embodied.envs.from_gymnasium import FromGymnasium

WrapperObsType = TypeVar("WrapperObsType")

# # Action meanings
# actions = {
#     0: 'left',    # Rotate left
#     1: 'right',   # Rotate right
#     2: 'forward', # Move forward
#     3: 'pickup',  # Pick up an object
#     4: 'drop',    # Drop an object
#     5: 'toggle',  # Toggle/open/close an object (e.g., doors)
#     6: 'done'     # No-op
# }

# Observation space (in batch_env class)
# ['image', 'reward', 'intrinsic_reward', 'is_first', 'is_last', 'is_terminal']
#  {'image': Space(dtype=uint8, shape=(64, 64, 3), low=0, high=255),
#   'reward': Space(dtype=float32, shape=(), low=-inf, high=inf),
#   'intrinsic_reward': Space(dtype=float32, shape=(1,), low=-inf, high=inf),
#   'is_first': Space(dtype=bool, shape=(), low=False, high=True),
#   'is_last': Space(dtype=bool, shape=(), low=False, high=True),
#   'is_terminal': Space(dtype=bool, shape=(), low=False, high=True)}

# Action space (in batch_env class)
# {'action': Space(dtype=float32, shape=(7,), low=0, high=1),
# 'reset': Space(dtype=bool, shape=(), low=False, high=True)}

# Step
# Q: when is terminated and last?
# A: terminated is when the episode is done; last is when the episode is done and the last state is reached


class HideMission(ObservationWrapper):
    """Remove the 'mission' string from the observation."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = cast(gymnasium.spaces.Dict, self.observation_space)
        obs_space.spaces.pop("mission")

    def observation(self, observation: dict):
        observation.pop("mission")
        return observation
    

class ObsPlusAgentPosAndDirWrapper(ObservationWrapper):
    """
    Returns the original observation plus the agent's position and direction.
    The observation is a dictionary following the original format, but with the agent's position and direction added.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = ObsPlusAgentPosAndDirWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs
        {'image': ..., 'direction': 0, 'mission': 'grand mission', 'agent_pos_and_dir': array([0, 0, 0], dtype=int32)}
    """

    def __init__(self, env):
        super().__init__(env)
        
        # Define the new observation space using Dict
        self.observation_space = env.observation_space
        self.observation_space.spaces["agent_pos_and_dir"] = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.int32),  # Lower bounds for x, y, dir
            high=np.array([env.unwrapped.width, env.unwrapped.height, 3], dtype=np.int32),  # Upper bounds
            shape=(3,),  # (x, y, dir)
            dtype=np.int32
        )


    def observation(self, obs):
        """Returns the original observation plus the agent's position and direction."""
        obs["agent_pos_and_dir"] = np.array([
                int(self.unwrapped.agent_pos[0]),  # x-coordinate
                int(self.unwrapped.agent_pos[1]),  # y-coordinate
                int(self.unwrapped.agent_dir)
            ], dtype=np.int32)

        return obs


class WrappedMinigrid(FromGymnasium):
    def __init__(self, task: str, fully_observable: bool, pixel_input: bool, hide_mission: bool):
        env = gymnasium.make(f"MiniGrid-{task}-v0", render_mode="rgb_array")
        if fully_observable:
            if pixel_input:
                env = RGBImgObsWrapper(env)
                env = HideMission(env)
                env = ObsPlusAgentPosAndDirWrapper(env)
                # env = RGBImgPartialObsWrapper(env)
            else:
                env = FullyObsWrapper(env)
                env = ImgObsWrapper(env)
                env = ObsPlusAgentPosAndDirWrapper(env)
        else:
            if pixel_input:
                env = RGBImgPartialObsWrapper(env)
                env = HideMission(env)
                env = ObsPlusAgentPosAndDirWrapper(env)
            else:
                env = ImgObsWrapper(env)
                env = ObsPlusAgentPosAndDirWrapper(env)
        # if hide_mission:
        #     env = HideMission(env)
        super().__init__(env=env)


# also wrap in ResizeImage so that we can handle size kwarg
class Minigrid(ResizeImage):
    def __init__(self, *args, size, **kwargs):
        super().__init__(WrappedMinigrid(*args, **kwargs), size=size)
