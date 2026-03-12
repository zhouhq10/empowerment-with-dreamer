"""Environment definitions for the empowerment experiments.

Two independent environment implementations are provided:

**MiniGrid-based environments** (used for the main experiments):

- :class:`Ice` – custom ``WorldObj`` tile that causes the agent to slip to a
  random neighbouring cell on any movement action.
- :class:`MiniGridEnvWithIce` – subclass of ``MiniGridEnv`` that overrides
  ``step()`` to apply ice-slip dynamics.
- :class:`MixedEnv` – the specific 12×12 layout used in the experiments
  (ice tiles, lava, walls, a goal, and a fixed agent start position).
- :class:`AgentPosAndDirWrapper` – observation wrapper that reduces the full
  MiniGrid observation to a compact ``(x, y, dir)`` tuple suitable for
  tabular agents.
- :func:`get_all_states` – enumerate every valid ``(x, y, dir)`` state.
- :func:`get_ground_truth_transition_probabilities` – analytically compute
  the true ``P(s'|s, a)`` matrix for ``MixedEnv``.

**Standalone GridWorld** (used for early-stage prototyping):

- :class:`StateType` – enum for the tile types in the custom grid world.
- :class:`GridWorld` – simple grid-world with deterministic, stochastic (ice),
  death, obstacle, and portal tile types.
"""

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


# Register the custom "ice" object type with MiniGrid's global registry so
# that it can be encoded/decoded consistently alongside built-in types.
if "ice" not in OBJECT_TO_IDX:
    next_index = len(OBJECT_TO_IDX)
    OBJECT_TO_IDX["ice"] = next_index
    IDX_TO_OBJECT[next_index] = "ice"


class Ice(WorldObj):
    """A traversable tile that causes the agent to slip (stochastic movement).

    The tile is rendered as a solid blue rectangle.  Any movement action
    performed while standing on an ``Ice`` tile sends the agent to a uniformly
    random non-wall neighbouring cell rather than applying the intended action.
    """

    def __init__(self) -> None:
        super().__init__('ice', 'blue')  # type="ice", color="blue"

    def can_overlap(self) -> bool:
        # The agent can step onto this tile
        return True

    def render(self, img: np.ndarray) -> None:
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class MiniGridEnvWithIce(MiniGridEnv):
    """MiniGrid environment extended with stochastic ice-tile dynamics.

    Movement actions performed while the agent stands on an :class:`Ice` tile
    are ignored: instead the agent slips to a uniformly random non-wall
    neighbouring cell.  All other MiniGrid mechanics (pickup, drop, toggle,
    etc.) are unchanged.
    """

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Position and cell in front of the agent (used for forward action)
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        is_movement_action = (
            action == self.actions.left
            or action == self.actions.right
            or action == self.actions.forward
        )
        current_cell = self.grid.get(*self.agent_pos)

        if is_movement_action:
            if not current_cell or current_cell.type != "ice":
                # --- Normal (non-ice) movement: standard MiniGrid logic ---
                if action == self.actions.left:
                    self.agent_dir -= 1
                    if self.agent_dir < 0:
                        self.agent_dir += 4

                elif action == self.actions.right:
                    self.agent_dir = (self.agent_dir + 1) % 4

                elif action == self.actions.forward:
                    if fwd_cell is None or fwd_cell.can_overlap():
                        self.agent_pos = tuple(fwd_pos)
                    if fwd_cell is not None and fwd_cell.type == "goal":
                        terminated = True
                        reward = self._reward()
                    if fwd_cell is not None and fwd_cell.type == "lava":
                        terminated = True
            else:
                # --- Ice slip: move to a uniformly random walkable neighbour ---
                x, y = self.agent_pos
                neighbors = []
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        neighbor_obj = self.grid.get(nx, ny)
                        if neighbor_obj is None or neighbor_obj.can_overlap():
                            neighbors.append((nx, ny))

                if neighbors:
                    self.agent_pos = tuple(neighbors[self.np_random.choice(len(neighbors))])

                # Check terminal conditions after slipping
                if (
                    self.grid.get(*self.agent_pos) is not None
                    and self.grid.get(*self.agent_pos).type == "lava"
                ):
                    terminated = True

                if (
                    self.grid.get(*self.agent_pos) is not None
                    and self.grid.get(*self.agent_pos).type == "goal"
                ):
                    terminated = True
                    reward = self._reward()

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop a carried object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle / activate an object
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
    """The 12×12 mixed-dynamics environment used in the experiments.

    The layout contains ice tiles (stochastic), lava (terminal), inner walls,
    and a goal tile in the lower-right corner.  The agent starts at a fixed
    position facing left (direction 3).

    Args:
        size: Side length of the square grid (default 12).
        agent_start_pos: Initial ``(x, y)`` position of the agent.
        agent_start_dir: Initial facing direction (0=right, 1=down,
            2=left, 3=up).
        max_steps: Episode step limit.  Defaults to 10 000 to rarely
            truncate episodes in this challenging environment.
    """

    def __init__(
        self,
        size: int = 12,
        agent_start_pos: tuple[int, int] = (10, 4),
        agent_start_dir: int = 3,
        max_steps: int | None = None,
        **kwargs,
    ) -> None:
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            # Large value to rarely truncate; inf is not allowed (needs int)
            max_steps = int(10000)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs
        )

    @staticmethod
    def _gen_mission() -> str:
        return "grand mission"

    def _gen_grid(self, width: int, height: int) -> None:
        # Create an empty grid with surrounding walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place ice tiles (stochastic movement)
        ice_positions = [
            (1, 1), (1, 6), (3, 6), (4, 4), (4, 5), (5, 5), (4, 6), (5, 6),
            (4, 7), (4, 8), (6, 4), (6, 5), (8, 9), (5, 1), (8, 2), (8, 3),
            (9, 2), (9, 3),
        ]
        for x, y in ice_positions:
            self.put_obj(Ice(), x, y)

        # Place lava tiles (terminal on entry)
        lava_positions = [
            (2, 10), (3, 7), (5, 4), (9, 7), (4, 1), (5, 2), (6, 6), (1, 8)
        ]
        for x, y in lava_positions:
            self.put_obj(Lava(), x, y)

        # Place inner wall tiles (block movement)
        wall_positions = [
            (7, 4), (7, 3), (7, 2), (7, 5), (8, 5), (9, 5), (10, 5)
        ]
        for x, y in wall_positions:
            self.put_obj(Wall(), x, y)

        # Goal tile in the lower-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place agent at fixed start position or randomly
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


class AgentPosAndDirWrapper(ObservationWrapper):
    """Observation wrapper that returns only the agent's ``(x, y, dir)`` state.

    Reduces the full MiniGrid image observation to a compact tuple suitable
    for tabular (count-based) agents.

    Example::

        >>> env = AgentPosAndDirWrapper(MixedEnv())
        >>> obs, _ = env.reset()
        >>> obs
        (10, 4, 3)   # starting position and direction
    """

    def __init__(self, env: MiniGridEnv) -> None:
        super().__init__(env)

        # Flat discrete observation: (x, y, dir)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(env.unwrapped.width),
            spaces.Discrete(env.unwrapped.height),
            spaces.Discrete(4),
        ))

    def observation(self, obs: ObsType) -> tuple[int, int, int]:
        """Return the agent's position and direction.

        Args:
            obs: Original MiniGrid observation (ignored).

        Returns:
            ``(x, y, dir)`` tuple.
        """
        return (
            int(self.unwrapped.agent_pos[0]),
            int(self.unwrapped.agent_pos[1]),
            int(self.unwrapped.agent_dir),
        )


def get_all_states(env: MiniGridEnv) -> list[tuple[int, int, int]]:
    """Return all valid ``(x, y, dir)`` states (i.e. non-wall cells × 4 dirs).

    Args:
        env: An unwrapped MiniGrid environment whose grid has been initialised
            (i.e. ``env.reset()`` has been called at least once).

    Returns:
        List of ``(x, y, dir)`` tuples.
    """
    states = []
    for x in range(env.width):
        for y in range(env.height):
            if env.grid.get(x, y) is None or env.grid.get(x, y).type != "wall":
                for dir in range(4):
                    states.append((x, y, dir))
    return states


def get_ground_truth_transition_probabilities(
    env: MiniGridEnvWithIce, num_actions: int = 3
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Analytically compute the ground-truth ``P(s'|s, a)`` matrix.

    Handles the two dynamics implemented in :class:`MiniGridEnvWithIce`:

    * **Normal tiles** – deterministic left/right turns and forward movement
      (blocked by walls).
    * **Ice tiles** – all three movement actions slip the agent to a uniformly
      random walkable neighbour; the agent's facing direction is preserved.

    Lava and goal states are treated as absorbing: all actions loop back to
    the same state (the agent is assumed never to act after episode end).

    Args:
        env: Unwrapped ``MiniGridEnvWithIce`` instance (e.g. ``MixedEnv``).
            ``env.reset()`` must have been called before this function.
        num_actions: Expected number of actions (default 3: left, right, forward).

    Returns:
        P: Transition matrix of shape ``(|S|, num_actions, |S|)``.
        terminal_transitions: Boolean array of shape ``(|S|, num_actions, |S|)``
            indicating which transitions end the episode.
        state_to_idx: Mapping from ``(x, y, dir)`` tuples to matrix indices.
    """
    if not hasattr(env, 'grid') or env.grid is None:
        raise ValueError("Environment grid not initialized. Call env.reset() first.")

    all_possible_states = get_all_states(env.unwrapped)
    num_states = len(all_possible_states)

    state_to_idx: dict[tuple[int, int, int], int] = {
        s: i for i, s in enumerate(all_possible_states)
    }

    P = np.zeros((num_states, num_actions, num_states))
    terminal_transitions = np.zeros((num_states, num_actions, num_states), dtype=bool)

    # Any transition that lands on lava or goal is terminal
    for idx_s, state in enumerate(all_possible_states):
        x, y, agent_dir = state
        current_cell = env.grid.get(x, y)
        if current_cell is not None and current_cell.type in ("lava", "goal"):
            terminal_transitions[:, :, idx_s] = True

    # Simulate transitions for each (state, action) pair
    for idx_s, state in enumerate(all_possible_states):
        x, y, agent_dir = state
        current_cell = env.grid.get(x, y)
        is_on_ice = (current_cell is not None and current_cell.type == "ice")

        if is_on_ice:
            # --- Ice: uniform slip to any walkable neighbour ---
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.width and 0 <= ny < env.height:
                    neighbor_cell = env.grid.get(nx, ny)
                    if neighbor_cell is None or neighbor_cell.can_overlap():
                        neighbors.append((nx, ny))

            if not neighbors:
                # Defensively handle fully-enclosed cell: agent stays put
                slip_prob = 1.0
                neighbors = [(x, y)]
            else:
                slip_prob = 1.0 / len(neighbors)

            for action_idx in range(3):  # 0:left, 1:right, 2:forward
                for next_x, next_y in neighbors:
                    # Slipping preserves the original facing direction
                    next_state = (next_x, next_y, agent_dir)
                    if next_state in state_to_idx:
                        idx_s_prime = state_to_idx[next_state]
                        P[idx_s, action_idx, idx_s_prime] += slip_prob

        else:
            # --- Deterministic transitions ---
            if current_cell is not None and current_cell.type == "lava":
                # Lava is absorbing: every action loops back to the same state
                for action_idx in range(num_actions):
                    P[idx_s, action_idx, idx_s] = 1.0
            else:
                # Action 0: Turn left (counter-clockwise)
                next_dir_left = (agent_dir - 1 + 4) % 4
                next_state_left = (x, y, next_dir_left)
                if next_state_left in state_to_idx:
                    P[idx_s, 0, state_to_idx[next_state_left]] = 1.0

                # Action 1: Turn right (clockwise)
                next_dir_right = (agent_dir + 1) % 4
                next_state_right = (x, y, next_dir_right)
                if next_state_right in state_to_idx:
                    P[idx_s, 1, state_to_idx[next_state_right]] = 1.0

                # Action 2: Move forward
                # Directions: 0=right (+x), 1=down (+y), 2=left (−x), 3=up (−y)
                fwd_dx, fwd_dy = 0, 0
                if agent_dir == 0:
                    fwd_dx = 1
                elif agent_dir == 1:
                    fwd_dy = 1
                elif agent_dir == 2:
                    fwd_dx = -1
                elif agent_dir == 3:
                    fwd_dy = -1

                fwd_pos = (x + fwd_dx, y + fwd_dy)
                fwd_cell = env.grid.get(*fwd_pos)

                if fwd_cell is None or fwd_cell.can_overlap():
                    # Walkable tile: move forward
                    next_state_fwd = (fwd_pos[0], fwd_pos[1], agent_dir)
                else:
                    # Blocked (wall, closed door, etc.): stay in place
                    next_state_fwd = (x, y, agent_dir)

                if next_state_fwd in state_to_idx:
                    P[idx_s, 2, state_to_idx[next_state_fwd]] = 1.0

    # Sanity check: each (state, action) row should sum to 1
    sum_probs = P.sum(axis=2)
    if not np.allclose(sum_probs[sum_probs > 0], 1.0):
        print("Warning: Some transition probabilities do not sum to 1.")

    return P, terminal_transitions, state_to_idx


############################################################################################################
### Own GridWorld implementation ###########################################################################
############################################################################################################


class StateType(Enum):
    """Tile types for the custom :class:`GridWorld` environment."""
    NORMAL = "normal"
    OBSTACLE = "obstacle"
    ICE = "ice"
    DEATH = "death"
    PORTAL = "portal"


class GridWorld:
    """Simple tabular grid world with several tile types.

    Tile semantics:

    * **NORMAL** – deterministic movement; blocked by ``OBSTACLE`` tiles.
    * **OBSTACLE** – impassable; agent cannot enter (acts as wall).
    * **ICE** – stochastic: all actions lead to a uniformly random neighbour.
    * **DEATH** – absorbing: all actions loop back to the same tile; episode
      terminates when the agent reaches a death tile.
    * **PORTAL** – wraps around the grid (must be on the border).

    Args:
        width: Grid width.
        height: Grid height.
        state_types: Mapping from ``(x, y)`` to :class:`StateType`.  Missing
            entries default to ``NORMAL``.
        random_seed: Optional seed for reproducible transitions.
    """

    def __init__(
        self,
        width: int,
        height: int,
        state_types: Dict[Tuple[int, int], StateType] | None = None,
        random_seed: int | None = None,
    ) -> None:
        if random_seed is not None:
            import random as _random
            _random.seed(random_seed)
            np.random.seed(random_seed)

        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4  # 0:up, 1:right, 2:down, 3:left

        self.agent_pos: tuple[int, int] | None = None

        # Fill in NORMAL for any position not explicitly specified
        self.state_types = state_types or {
            (x, y): StateType.NORMAL
            for x in range(width)
            for y in range(height)
        }
        if len(self.state_types) <= self.num_states:
            for x in range(width):
                for y in range(height):
                    if (x, y) not in self.state_types:
                        self.state_types[(x, y)] = StateType.NORMAL

        # Build convenience sets for each tile type
        self.obstacles = {pos for pos, t in self.state_types.items() if t == StateType.OBSTACLE}
        self.ice_states = {pos for pos, t in self.state_types.items() if t == StateType.ICE}
        self.death_states = {pos for pos, t in self.state_types.items() if t == StateType.DEATH}
        self.portals = {pos for pos, t in self.state_types.items() if t == StateType.PORTAL}
        self.deterministic_states = {
            pos for pos, t in self.state_types.items() if t == StateType.NORMAL
        }

        self.transition_prob = self._create_transition_prob()
        validate_probability_distribution(self.transition_prob, axis=2)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[int, int]:
        """Place the agent at a random non-obstacle position.

        Returns:
            Initial ``(x, y)`` position.
        """
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if (x, y) not in self.obstacles:
                self.agent_pos = (x, y)
                break
        return self.agent_pos

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, bool, dict]:
        """Advance the environment by one step.

        Args:
            action: Integer action (0:up, 1:right, 2:down, 3:left).

        Returns:
            obs: New agent position ``(x, y)``.
            reward: Always 0 (no extrinsic reward).
            terminated: ``True`` if the agent entered a death tile.
            truncated: Always ``False``.
            info: Empty dict.
        """
        current_state = self._get_state_index(self.agent_pos[0], self.agent_pos[1])
        prob_distribution = self.transition_prob[current_state, action]
        next_state = np.random.choice(self.num_states, p=prob_distribution)
        self.agent_pos = self._get_coordinates(next_state)

        obs = self.agent_pos
        reward = 0  # Fixed to zero for now; current env has no extrinsic rewards
        terminated = False
        truncated = False
        info: dict = {}

        if self.agent_pos in self.death_states:
            terminated = True

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_at_wall(self, x: int, y: int) -> bool:
        """Return ``True`` if ``(x, y)`` is on the grid border."""
        return x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1

    def _create_transition_prob(self) -> np.ndarray:
        """Build the full ``(|S|, |A|, |S|)`` transition probability matrix.

        Tile types are processed in order: NORMAL → ICE → DEATH/OBSTACLE →
        PORTAL.  Each type fills in the rows of ``trans_prob`` corresponding
        to states of that type.

        Returns:
            Array of shape ``(num_states, num_actions, num_states)``.
        """
        trans_prob = np.zeros((self.num_states, self.num_actions, self.num_states))

        # Assert each cell has exactly one type
        assert len(
            reduce(
                set.union,
                [self.obstacles, self.ice_states, self.death_states, self.portals, self.deterministic_states]
            )
        ) == self.num_states, "Some states seem to have multiple types, please fix the state_types dict"

        for (x, y), state_type in self.state_types.items():
            if state_type == StateType.NORMAL:
                # Deterministic movement; blocked by obstacles (stay in place)
                current_state = self._get_state_index(x, y)
                for action in range(self.num_actions):
                    next_x, next_y = self._get_next_position(x, y, action)
                    next_state = self._get_state_index(next_x, next_y)
                    if (next_x, next_y) in self.obstacles:
                        next_state = current_state
                    trans_prob[current_state, action, next_state] = 1.0

            elif state_type == StateType.ICE:
                # Each action leads to a uniformly random adjacent state
                current_state = self._get_state_index(x, y)
                outcomes = []
                for action in range(self.num_actions):
                    next_x, next_y = self._get_next_position(x, y, action)
                    next_state = self._get_state_index(next_x, next_y)
                    if (next_x, next_y) in self.obstacles:
                        next_state = current_state
                    outcomes.append(next_state)

                prob = 1.0 / len(outcomes)
                for action in range(self.num_actions):
                    for next_state in outcomes:
                        trans_prob[current_state, action, next_state] += prob

            elif state_type == StateType.DEATH or state_type == StateType.OBSTACLE:
                # Absorbing: all actions loop back to the same state
                current_state = self._get_state_index(x, y)
                for action in range(self.num_actions):
                    trans_prob[current_state, action, current_state] = 1.0

            elif state_type == StateType.PORTAL:
                # Wrap-around portals; only valid on the grid border
                current_state = self._get_state_index(x, y)
                for action in range(self.num_actions):
                    if self._is_at_wall(x, y):
                        next_x, next_y = x, y
                        if action == 0:   # up
                            next_y = 0 if y == self.height - 1 else min(self.height - 1, y + 1)
                        elif action == 1: # right
                            next_x = 0 if x == self.width - 1 else min(self.width - 1, x + 1)
                        elif action == 2: # down
                            next_y = self.height - 1 if y == 0 else max(0, y - 1)
                        elif action == 3: # left
                            next_x = self.width - 1 if x == 0 else max(0, x - 1)

                        next_state = self._get_state_index(next_x, next_y)
                        if (next_x, next_y) in self.obstacles:
                            next_state = current_state
                        trans_prob[current_state, action, next_state] = 1.0
                    else:
                        raise ValueError("Portal states must be at the edge of the grid")

        return trans_prob

    def _get_next_position(self, x: int, y: int, action: int) -> tuple[int, int]:
        """Compute the intended next position for *action* from ``(x, y)``.

        Clamps to grid boundaries (no wrapping for non-portal tiles).

        Args:
            x: Current column.
            y: Current row.
            action: 0:up, 1:right, 2:down, 3:left.

        Returns:
            ``(next_x, next_y)`` after applying *action*.
        """
        next_x, next_y = x, y

        if action == 0:   # up
            next_y = min(self.height - 1, y + 1)
        elif action == 1: # right
            next_x = min(self.width - 1, x + 1)
        elif action == 2: # down
            next_y = max(0, y - 1)
        elif action == 3: # left
            next_x = max(0, x - 1)

        return next_x, next_y

    def _get_state_index(self, x: int, y: int) -> int:
        """Map ``(x, y)`` to a flat state index (row-major)."""
        return y * self.width + x

    def _get_coordinates(self, state: int) -> tuple[int, int]:
        """Inverse of :meth:`_get_state_index`."""
        return state % self.width, state // self.width
