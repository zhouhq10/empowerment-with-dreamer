from __future__ import annotations

from enum import Enum
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
from functools import reduce
from typing import Tuple, Dict, Any, SupportsFloat

from utils import validate_probability_distribution

from gymnasium.core import ActType, ObsType
from gymnasium import spaces

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Lava, Wall, Goal
from minigrid.utils.rendering import fill_coords, point_in_rect
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT, COLORS

from minigrid.wrappers import ObservationWrapper


############################################################################################################
### Modified MiniGrid environment ##########################################################################
############################################################################################################


# Register "ice" state type
if "ice" not in OBJECT_TO_IDX:
    next_index = len(OBJECT_TO_IDX)
    OBJECT_TO_IDX["ice"] = next_index
    IDX_TO_OBJECT[next_index] = "ice"

class Ice(WorldObj):
    def __init__(self):
        super().__init__('ice', 'blue')  # type="ice", color="blue"

    def can_overlap(self):
        # the agent can move onto this tile
        return True
    
    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class MiniGridEnvWithIce(MiniGridEnv):
    """ A MiniGrid environment with "ice" tiles where transitions are stochastic. """    
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        is_movement_action = (action == self.actions.left or action == self.actions.right or action == self.actions.forward)
        current_cell = self.grid.get(*self.agent_pos)
        if is_movement_action:
            if not current_cell or current_cell.type != "ice": # if the agent is not on ice (cell = None is regular cell), move normally
                # code from usual MiniGridEnv .step() function
                # Rotate left
                if action == self.actions.left:
                    self.agent_dir -= 1
                    if self.agent_dir < 0:
                        self.agent_dir += 4

                # Rotate right
                elif action == self.actions.right:
                    self.agent_dir = (self.agent_dir + 1) % 4

                # Move forward
                elif action == self.actions.forward:
                    if fwd_cell is None or fwd_cell.can_overlap():
                        self.agent_pos = tuple(fwd_pos)
                    if fwd_cell is not None and fwd_cell.type == "goal":
                        terminated = True
                        reward = self._reward()
                    if fwd_cell is not None and fwd_cell.type == "lava":
                        terminated = True
            else: # If the agent is on ice, slip to random neighboring cell
                # Gather valid neighbors (non-wall, non-locked-door spaces)
                x, y = self.agent_pos
                neighbors = []
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        neighbor_obj = self.grid.get(nx, ny)
                        if neighbor_obj is None or neighbor_obj.can_overlap():
                            neighbors.append((nx, ny))
                
                # Slip to random neighbor if possible
                if neighbors:
                    self.agent_pos = tuple(neighbors[self.np_random.choice(len(neighbors))])

                # if the agent slipped onto a lava tile, terminate the episode
                if self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == "lava":
                    terminated = True

                # if the agent slipped onto a goal tile, terminate the episode
                if self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == "goal":
                    terminated = True
                    reward = self._reward()

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

class MixedEnv(MiniGridEnvWithIce):
    def __init__(
        self,
        size=12,
        agent_start_pos=(10, 4),
        agent_start_dir=3,
        max_steps=None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            # set to a large number to only rarely truncate because the environment is difficult, and 
            # instead only care about termination (inf is not allowed, need integer)
            max_steps = int(10000) 

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place ice tiles
        ice_positions = [
            (1, 1), (1, 6), (3, 6), (4, 4), (4, 5), (5, 5), (4, 6), (5, 6), (4, 7), (4, 8), (6, 4), (6, 5), (8, 9), (5, 1), (8, 2), (8, 3), (9, 2), (9, 3)
        ]
        for x, y in ice_positions:
            self.put_obj(Ice(), x, y)

        # Place lava tiles
        lava_positions = [
            (2, 10), (3, 7), (5, 4), (9, 7), (4, 1), (5, 2), (6, 6), (1, 8)
        ]
        for x, y in lava_positions:
            self.put_obj(Lava(), x, y)

        # Place wall tiles
        wall_positions = [
            (7, 4), (7, 3), (7, 2), (7, 5), (8, 5), (9, 5), (10, 5)
        ]
        for x, y in wall_positions:
            self.put_obj(Wall(), x, y)

        # Place goal tile in lower right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"

class AgentPosAndDirWrapper(ObservationWrapper):
    """
    Returns only the agent's position and direction as the observation.
    The observation is a tuple of (x, y, dir) where x, y are the coordinates and dir is the orientation.
    
    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import AgentPosWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = AgentPosAndDirWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs
        (1, 1, 0)  # starting in the top-left corner facing right
    """

    def __init__(self, env):
        super().__init__(env)
        
        # Define the new observation space using Tuple
        self.observation_space = spaces.Tuple((
            spaces.Discrete(env.unwrapped.width),
            spaces.Discrete(env.unwrapped.height),
            spaces.Discrete(4)
        ))

    def observation(self, obs):
        """Returns the agent's position as a tuple (x, y)."""
        return (
            int(self.unwrapped.agent_pos[0]), 
            int(self.unwrapped.agent_pos[1]),
            int(self.unwrapped.agent_dir)
            )
    
# get all possible states in the environment (all except walls)
def get_all_states(env):
    states = []
    for x in range(env.width):
        for y in range(env.height):
            if env.grid.get(x, y) is None or env.grid.get(x, y).type != "wall":
                for dir in range(4):
                    states.append((x, y, dir))
    return states

def get_ground_truth_transition_probabilities(env: MiniGridEnvWithIce, num_actions=3):
    """
    Computes the ground truth transition probability matrix P(s'|s, a) 
    for a MiniGrid environment, specifically handling walls and ice tiles
    as defined in MiniGridEnvWithIce. Assumes the state is (x, y, dir)
    and actions are 0:left, 1:right, 2:forward.

    Args:
        env: An instance of the MiniGridEnvWithIce environment (e.g., MixedEnv wrapped 
             with AgentPosAndDirWrapper BUT passed as env.unwrapped). It's assumed env.reset() has been 
             called at least once to initialize the grid.
        num_actions: The number of actions in the environment (should be 3 for left, right, forward).

    Returns:
        tuple: (
            P (np.ndarray): The transition probability matrix of shape 
                            (|S|, |A|, |S|), where |S| is the number of 
                            valid states and |A| is the number of actions (3).
            terminal_transitions (np.ndarray): A matrix of shape (|S|, |A|, |S|) 
                            indicating which transitions terminate the episode.
            state_to_idx (dict): A mapping from state tuples (x, y, dir) to 
                                 their corresponding index in the P matrix.
            idx_to_state (list): A list where the index corresponds to the
                                 index in P, and the value is the state tuple.
        )
    """
    if not hasattr(env, 'grid') or env.grid is None:
         raise ValueError("Environment grid not initialized. Call env.reset() first.")

    # Get all valid states (x, y, dir) where (x, y) is not a wall
    all_possible_states = get_all_states(env.unwrapped)
    num_states = len(all_possible_states)
    
    # Create state-to-index mapping
    state_to_idx = {s: i for i, s in enumerate(all_possible_states)}

    # Initialize transition probability matrix P[s, a, s']
    P = np.zeros((num_states, num_actions, num_states))
    # Initialize terminal transitions matrix
    terminal_transitions = np.zeros((num_states, num_actions, num_states), dtype=bool)

    # Fill the terminal transitions matrix
    for idx_s, state in enumerate(all_possible_states):
        x, y, agent_dir = state
        current_cell = env.grid.get(x, y)
        if current_cell is not None and (current_cell.type == "lava" or current_cell.type == "goal"):
            # All transitions that end in lava or goal are terminal, regardless of the state and action that lead there
            terminal_transitions[:, :, idx_s] = True


    # --- Simulate transitions for each state and action ---
    for idx_s, state in enumerate(all_possible_states):
        x, y, agent_dir = state
        current_cell = env.grid.get(x, y)
        is_on_ice = (current_cell is not None and current_cell.type == "ice")

        # --- Handle Ice Tile Transitions (Stochastic) ---
        if is_on_ice:
            # For ice, all movement actions (left, right, forward) lead to slipping
            
            # Find valid neighbor positions (non-wall, within grid)
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]: # Check right, left, up, down
                nx, ny = x + dx, y + dy
                # Check grid boundaries
                if 0 <= nx < env.width and 0 <= ny < env.height:
                    neighbor_cell = env.grid.get(nx, ny)
                    # Check if neighbor is walkable (None or can_overlap)
                    if neighbor_cell is None or neighbor_cell.can_overlap():
                        neighbors.append((nx, ny))
            
            # Calculate slip probability
            if not neighbors: 
                # Should not happen in typical grids unless trapped by walls,
                # but handle defensively: agent stays in the same state.
                slip_prob = 1.0 
                neighbors = [(x,y)] # Slip target is the current pos
            else:
                slip_prob = 1.0 / len(neighbors)

            # Apply slip probability to all movement actions
            for action_idx in range(3): # 0:left, 1:right, 2:forward
                for next_x, next_y in neighbors:
                    # Slipping keeps the *original* direction
                    next_state = (next_x, next_y, agent_dir) 
                    if next_state in state_to_idx: # Ensure the target state is valid
                        idx_s_prime = state_to_idx[next_state]
                        P[idx_s, action_idx, idx_s_prime] += slip_prob # Use += in case multiple neighbors map to same state (unlikely here)

        # --- Handle Deterministic Transitions ---
        else: 
            # if lava state, one could argue that all actions lead to the same state even though the agent never observes this
            if current_cell is not None and current_cell.type == "lava":
                # All actions lead to the same state (the lava state)
                for action_idx in range(num_actions):
                    P[idx_s, action_idx, idx_s] = 1.0
            else:
                # For normal states, handle left, right, and forward actions
                # Action 0: Turn Left
                next_dir_left = (agent_dir - 1 + 4) % 4
                next_state_left = (x, y, next_dir_left)
                if next_state_left in state_to_idx:
                    idx_s_prime_left = state_to_idx[next_state_left]
                    P[idx_s, 0, idx_s_prime_left] = 1.0
                # else: state is invalid (e.g., wall), probability remains 0 (shouldn't happen with get_all_states)

                # Action 1: Turn Right
                next_dir_right = (agent_dir + 1) % 4
                next_state_right = (x, y, next_dir_right)
                if next_state_right in state_to_idx:
                    idx_s_prime_right = state_to_idx[next_state_right]
                    P[idx_s, 1, idx_s_prime_right] = 1.0
                # else: state is invalid, probability remains 0

                # Action 2: Move Forward
                # Calculate position in front of the agent
                fwd_dx, fwd_dy = 0, 0
                if agent_dir == 0: # Right
                    fwd_dx = 1
                elif agent_dir == 1: # Down (Note: MiniGrid y increases downwards)
                    fwd_dy = 1
                elif agent_dir == 2: # Left
                    fwd_dx = -1
                elif agent_dir == 3: # Up
                    fwd_dy = -1
                
                fwd_pos = (x + fwd_dx, y + fwd_dy)

                # Check cell in front
                fwd_cell = env.grid.get(*fwd_pos)

                # Determine next state based on forward cell
                if fwd_cell is None or fwd_cell.can_overlap():
                    # Can move forward (empty, goal, lava, ice)
                    next_state_fwd = (fwd_pos[0], fwd_pos[1], agent_dir)
                else:
                    # Cannot move forward (wall, closed door, etc.) - stay in place
                    next_state_fwd = (x, y, agent_dir) # Stay in the current state

                # Assign probability
                if next_state_fwd in state_to_idx:
                    idx_s_prime_fwd = state_to_idx[next_state_fwd]
                    P[idx_s, 2, idx_s_prime_fwd] = 1.0
                # else: state is invalid, probability remains 0

    # --- Verification (Optional but recommended) ---
    # Check if probabilities sum to 1 for each (state, action) pair
    sum_probs = P.sum(axis=2)
    if not np.allclose(sum_probs[sum_probs > 0], 1.0): # Check only non-terminal/valid states
         print("Warning: Some transition probabilities do not sum to 1.")
         # Find problematic states/actions for debugging:
         # problematic_indices = np.where(np.abs(sum_probs - 1.0) > 1e-6)
         # print("Problematic (state_idx, action_idx):", list(zip(*problematic_indices)))

    return P, terminal_transitions, state_to_idx


############################################################################################################
### Own GridWorld implementation ###########################################################################
############################################################################################################

class StateType(Enum):
    NORMAL = "normal"
    OBSTACLE = "obstacle"
    ICE = "ice"
    DEATH = "death"
    PORTAL = "portal"


class GridWorld:
    def __init__(self, width: int, height: int, state_types: Dict[Tuple[int, int], StateType] = None, random_seed: int = None):
        """Initialize the grid world environment.
        
        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            state_types (dict): A dictionary mapping (x, y) positions to StateType values. If the dict does not contain a mapping for a position, it is considered a normal state. If None, all states are normal states.
            random_seed (int): The random seed to use for reproducibility.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4  # up, right, down, left
        
        self.agent_pos = None
        
        self.state_types = state_types or {(x, y): StateType.NORMAL 
                                           for x in range(width) 
                                           for y in range(height)}
        
        # if the state_types dict does not contain a mapping for a position, it is considered a normal state
        if len(self.state_types) <= self.num_states:
            for x in range(width):
                for y in range(height):
                    if (x, y) not in self.state_types:
                        self.state_types[(x, y)] = StateType.NORMAL
        
        # Create sets for each state type for easy access
        self.obstacles = {pos for pos, type_ in self.state_types.items() 
                          if type_ == StateType.OBSTACLE}
        self.ice_states = {pos for pos, type_ in self.state_types.items() 
                           if type_ == StateType.ICE}
        self.death_states = {pos for pos, type_ in self.state_types.items() 
                             if type_ == StateType.DEATH}
        self.portals = {pos for pos, type_ in self.state_types.items() 
                        if type_ == StateType.PORTAL}
        # all remaining states are normal deterministic states
        self.deterministic_states = {pos for pos, type_ in self.state_types.items()
                                     if type_ == StateType.NORMAL}
        
        self.transition_prob = self._create_transition_prob()
        
        validate_probability_distribution(self.transition_prob, axis=2)

    def _is_at_wall(self, x, y):
        """Check if a position is at the edge of the grid."""
        return x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1

    def _create_transition_prob(self):
        """Creates the transition probability matrix. First makes all states deterministic and adds obstacles, then ice states, then death states, then portals."""
        trans_prob = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        # Assert that no state has multiple types by checking that the union of all sets is the same size as the total number of states, otherwise tell the user to fix the state_types dict
        assert len(reduce(set.union, [self.obstacles, self.ice_states, self.death_states, self.portals, self.deterministic_states])) == self.num_states, "Some states seem to have multiple types, please fix the state_types dict"

        for (x, y), state_type in self.state_types.items():
            if state_type == StateType.NORMAL:
                current_state = self._get_state_index(x, y)
                for action in range(self.num_actions):
                    next_x, next_y = self._get_next_position(x, y, action)
                    next_state = self._get_state_index(next_x, next_y)
                    
                    if (next_x, next_y) in self.obstacles:
                        next_state = current_state
                        
                    trans_prob[current_state, action, next_state] = 1.0
                    
            # 2. Add ice states in which each action has random outcomes
            elif state_type == StateType.ICE:
                current_state = self._get_state_index(x, y)
                
                # Get all possible outcomes in this state (all adjacent states)
                outcomes = []
                for action in range(self.num_actions):
                    next_x, next_y = self._get_next_position(x, y, action)
                    next_state = self._get_state_index(next_x, next_y)
                    
                    if (next_x, next_y) in self.obstacles:
                        next_state = current_state
                    
                    outcomes.append(next_state)
                    
                # Assign equal probability to each outcome
                prob = 1.0 / len(outcomes)
                for action in range(self.num_actions):
                    for next_state in outcomes:
                        trans_prob[current_state, action, next_state] += prob
            
            # 3. Add death states in which each action leads to the same state (and do the same for obstacles, even though the agent should never reach them)
            elif state_type == StateType.DEATH or state_type == StateType.OBSTACLE:
                current_state = self._get_state_index(x, y)
                                
                for action in range(self.num_actions):
                    trans_prob[current_state, action, current_state] = 1.0
                
            # 4. Add portals (states next to the wall of the grid, through which the agent teleports to the other side)
            elif state_type == StateType.PORTAL:
                current_state = self._get_state_index(x, y)
                
                for action in range(self.num_actions):
                    if self._is_at_wall(x, y):
                        next_x, next_y = x, y
                        if action == 0:  # up
                            if y == self.height - 1: # agent is at the top wall and moves up, teleport to the bottom wall
                                next_y = 0
                            else:
                                next_y = min(self.height - 1, y + 1)
                        elif action == 1:  # right
                            if x == self.width - 1: # agent is at the right wall and moves right, teleport to the left wall
                                next_x = 0
                            else:
                                next_x = min(self.width - 1, x + 1)
                        elif action == 2:  # down
                            if y == 0:
                                next_y = self.height - 1 # agent is at the bottom wall and moves down, teleport to the top wall
                            else:
                                next_y = max(0, y - 1)
                        elif action == 3:  # left
                            if x == 0:
                                next_x = self.width - 1 # agent is at the left wall and moves left, teleport to the right wall
                            else:
                                next_x = max(0, x - 1)
                        
                        next_state = self._get_state_index(next_x, next_y)
                        
                        # If there is an obstacle at the other side of the portal, stay in the current state
                        if (next_x, next_y) in self.obstacles:
                            next_state = current_state
                            
                        trans_prob[current_state, action, next_state] = 1.0
                    else: 
                        # portal cannot be placed in the middle of the grid
                        raise ValueError("Portal states must be at the edge of the grid")
                
        return trans_prob              
                    
    
    def _get_next_position(self, x, y, action):
        next_x, next_y = x, y
        
        if action == 0:  # up
            next_y = min(self.height - 1, y + 1)
        elif action == 1:  # right
            next_x = min(self.width - 1, x + 1)
        elif action == 2:  # down
            next_y = max(0, y - 1)
        elif action == 3:  # left
            next_x = max(0, x - 1)
            
        return next_x, next_y

    def reset(self):
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if (x, y) not in self.obstacles:
                self.agent_pos = (x, y)
                break
        return self.agent_pos

    def step(self, action):
        current_state = self._get_state_index(self.agent_pos[0], self.agent_pos[1])
        prob_distribution = self.transition_prob[current_state, action]
        next_state = np.random.choice(self.num_states, p=prob_distribution)
        self.agent_pos = self._get_coordinates(next_state)
        
        obs = self.agent_pos
        reward = 0 # fix to zero for now, current env has no rewards
        terminated = False # no termination condition for now
        truncated = False # no truncation for now
        info = {} # no additional info for now
        
        if self.agent_pos in self.death_states:
            terminated = True
        
        return obs, reward, terminated, truncated, info 

    def _get_state_index(self, x, y):
        return y * self.width + x

    def _get_coordinates(self, state):
        return state % self.width, state // self.width