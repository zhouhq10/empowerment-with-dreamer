import itertools
import random

from enum import Enum
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from functools import reduce


class StateType(Enum):
    NORMAL = "normal"
    OBSTACLE = "obstacle"
    ICE = "ice"
    DEATH = "death"
    PORTAL = "portal"
    GOAL = "goal"


class GridWorld:
    def __init__(self, width, height, state_types=None, random_seed=None):
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

        self.state_types = state_types or {
            (x, y): StateType.NORMAL for x in range(width) for y in range(height)
        }

        # if the state_types dict does not contain a mapping for a position, it is considered a normal state
        if len(self.state_types) <= self.num_states:
            for x in range(width):
                for y in range(height):
                    if (x, y) not in self.state_types:
                        self.state_types[(x, y)] = StateType.NORMAL

        # Create sets for each state type for easy access
        self.obstacles = {
            pos
            for pos, type_ in self.state_types.items()
            if type_ == StateType.OBSTACLE
        }
        self.ice_states = {
            pos for pos, type_ in self.state_types.items() if type_ == StateType.ICE
        }
        self.death_states = {
            pos for pos, type_ in self.state_types.items() if type_ == StateType.DEATH
        }
        self.portals = {
            pos for pos, type_ in self.state_types.items() if type_ == StateType.PORTAL
        }
        self.goals = {
            pos for pos, type_ in self.state_types.items() if type_ == StateType.GOAL
        }
        # all remaining states are normal deterministic states
        self.deterministic_states = {
            pos for pos, type_ in self.state_types.items() if type_ == StateType.NORMAL
        }

        self.transition_prob = self._create_transition_prob()

    def _is_at_wall(self, x, y):
        """Check if a position is at the edge of the grid."""
        return x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1

    def _create_transition_prob(self):
        """Creates the transition probability matrix. First makes all states deterministic and adds obstacles, then ice states, then death states, then portals."""
        trans_prob = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Assert that no state has multiple types by checking that the union of all sets is the same size as the total number of states, otherwise tell the user to fix the state_types dict
        assert (
            len(
                reduce(
                    set.union,
                    [
                        self.obstacles,
                        self.ice_states,
                        self.death_states,
                        self.portals,
                        self.goals,
                        self.deterministic_states,
                    ],
                )
            )
            == self.num_states
        ), "Some states seem to have multiple types, please fix the state_types dict"

        for (x, y), state_type in self.state_types.items():
            if state_type == StateType.NORMAL:
                current_state = self._get_state_index(x, y)
                for action in range(self.num_actions):
                    next_x, next_y = self._get_next_position(x, y, action)
                    next_state = self._get_state_index(next_x, next_y)

                    if (next_x, next_y) in self.obstacles:
                        next_state = current_state

                    trans_prob[current_state, action, next_state] = 1.0
            # 1. Add obstacles
            elif state_type == StateType.OBSTACLE:
                pass  # TODO?

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

            # 3. Add death states in which each action leads to the same state
            elif state_type == StateType.DEATH:
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
                            if (
                                y == self.height - 1
                            ):  # agent is at the top wall and moves up, teleport to the bottom wall
                                next_y = 0
                            else:
                                next_y = min(self.height - 1, y + 1)
                        elif action == 1:  # right
                            if (
                                x == self.width - 1
                            ):  # agent is at the right wall and moves right, teleport to the left wall
                                next_x = 0
                            else:
                                next_x = min(self.width - 1, x + 1)
                        elif action == 2:  # down
                            if y == 0:
                                next_y = (
                                    self.height - 1
                                )  # agent is at the bottom wall and moves down, teleport to the top wall
                            else:
                                next_y = max(0, y - 1)
                        elif action == 3:  # left
                            if x == 0:
                                next_x = (
                                    self.width - 1
                                )  # agent is at the left wall and moves left, teleport to the right wall
                            else:
                                next_x = max(0, x - 1)

                        next_state = self._get_state_index(next_x, next_y)

                        # If there is an obstacle at the other side of the portal, stay in the current state
                        if (next_x, next_y) in self.obstacles:
                            next_state = current_state

                        trans_prob[current_state, action, next_state] = 1.0
                    else:
                        # portal cannot be placed in the middle of the grid
                        raise ValueError(
                            "Portal states must be at the edge of the grid"
                        )

            # 5. Add desitination
            elif state_type == StateType.GOAL:
                current_state = self._get_state_index(x, y)
                pass 
                # for action in range(self.num_actions): 
                #     next_env = self._sample_new_env() # TODO implement
                #     next_x, next_y = self._get_next_position(x, y, action)
            
            else:
                import ipdb; ipdb.set_trace()
                raise ValueError(f"Unknown state type: {state_type}")

        return trans_prob

    def _sample_new_env(self):
        pass
    
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
                self.agent_pos = [x, y]
                break
        return self.agent_pos.copy()

    def step(self, action):
        current_state = self._get_state_index(self.agent_pos[0], self.agent_pos[1])
        prob_distribution = self.transition_prob[current_state, action]
        next_state = np.random.choice(self.num_states, p=prob_distribution)
        self.agent_pos = list(self._get_coordinates(next_state))
        return self.agent_pos

    def _get_state_index(self, x, y):
        return y * self.width + x

    def _get_coordinates(self, state):
        return state % self.width, state // self.width
