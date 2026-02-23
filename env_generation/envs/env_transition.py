import sys

sys.path.append("..")

import argparse
import random
import itertools
from enum import Enum
from functools import reduce

import numpy as np
from scipy.spatial.distance import cdist
from queue import PriorityQueue

import matplotlib as mpl
import matplotlib.pyplot as plt

from env_generation.envs.grid import GridWorld, StateType

# TODO:
# 1. place starting position


class EnvTransition(object):
    def __init__(
        self,
        width: int,
        height: int,
        args: argparse.Namespace,
    ) -> None:
        self.width = width
        self.height = height
        self.num_states = width * height

        self.grid = np.zeros((height, width), dtype=str)
        self.state_positions = {
            "death": [],
            "ice": [],
            "obstacle": [],
            "portal": [],
            "goal": [],
        }

        self.args = args
        self.env_series = args.env_series

        self.border_cells = self._store_border_cells()
        self.corner_cells = [
            (0, 0),
            (0, height - 1),
            (width - 1, 0),
            (width - 1, height - 1),
        ]

    def _store_border_cells(self) -> list:
        """Store the coordinates of the border cells."""
        # Ensure there are enough border cells to sample
        grid_width, grid_height = self.width, self.height

        # Define the border cells
        border_cells = []

        # Top row
        border_cells.extend([(x, 0) for x in range(grid_width)])
        # Bottom row
        border_cells.extend([(x, grid_height - 1) for x in range(grid_width)])
        # Left column (excluding corners to avoid duplication)
        border_cells.extend([(0, y) for y in range(1, grid_height - 1)])
        # Right column (excluding corners to avoid duplication)
        border_cells.extend([(grid_width - 1, y) for y in range(1, grid_height - 1)])

        return border_cells

    def init_env(self):
        # TODO: think about starting from a universally simple environment as baseline
        # (e.g. grid with no special states at all)?
        pass

    def _occupied(self, x: int, y: int) -> bool:
        """Check if a cell is occupied by any special state."""
        return not self.grid[y, x] == "0"

    def _calcu_occupied_cells(self) -> int:
        """Calculate the number of occupied cells in the grid."""
        occupied_cells = reduce(lambda x, y: x + y, self.state_positions.values())
        return len(occupied_cells)

    def _display_grid(self) -> None:
        print("\n".join(" ".join(row) for row in self.grid))

    def _reset_grid(self) -> None:
        """Reset the grid and state positions."""
        self.grid.fill("0")  # Reset grid to empty
        for state in self.state_positions:
            self.state_positions[state] = []

    def _generate_random_area(self, area_width: int, area_height: int) -> list:
        """Generate a random area of a given size within the grid."""
        # Ensure the area fits within the grid
        grid_width, grid_height = self.width, self.height
        if area_width > grid_width or area_height > grid_height:
            raise ValueError("The area size exceeds the grid size.")

        # Calculate the range for the top-left corner of the area
        max_x = grid_width - area_width
        max_y = grid_height - area_height

        # Randomly select the top-left corner
        top_left_x = random.randint(0, max_x)
        top_left_y = random.randint(0, max_y)

        # Define the selected area
        area_coordinates = [
            [top_left_x + x, top_left_y + y]
            for x in range(area_width)
            for y in range(area_height)
        ]

        return area_coordinates

    def _place_square_area(
        self, state_type: str, area_width: int, area_height: int
    ) -> None:
        """Place a square area of a given size within the grid."""
        area_coordinates = self._generate_random_area(area_width, area_height)
        for x, y in area_coordinates:
            if not self._occupied(x, y):
                self.grid[y, x] = state_type[0].upper()
                self.state_positions[state_type].append((x, y))

    def _place_constrained_state(
        self, state_type: str, num: int, pos_range: str = "last_column"
    ) -> None:
        """Place a given number of special states in the grid."""
        # TODO: need refactor to avoid hardcoding the state types
        if pos_range == "last_column":
            assert num <= self.height, f"Number of {state_type} exceeds height of grid."

            for _ in range(num):
                while True:
                    x, y = self.width - 1, random.randint(0, self.height - 1)
                    if not self._occupied(x, y):  # Empty cell
                        self.grid[y, x] = state_type[0].upper()
                        self.state_positions[state_type].append((x, y))
                        break
        elif pos_range == "random":
            num_normal_state = self.num_states - self._calcu_occupied_cells()
            assert (
                num <= num_normal_state
            ), f"Number of {state_type} states exceeds number of normal states."

            for _ in range(num):
                while True:
                    x, y = random.randint(0, self.width - 1), random.randint(
                        0, self.height - 1
                    )
                    if not self._occupied(x, y):  # Empty cell
                        self.grid[y, x] = state_type[0].upper()
                        self.state_positions[state_type].append((x, y))
                        break
        elif pos_range == "border":
            assert num <= 2 * (
                self.width + self.height - 2
            ), f"Number of {state_type} exceeds border length of grid."
            for _ in range(num):
                while True:
                    [(x, y)] = random.sample(self.border_cells, 1)
                    if not self._occupied(x, y):
                        self.grid[y, x] = state_type[0].upper()
                        self.state_positions[state_type].append((x, y))
                        break
        else:
            raise NotImplementedError(f"Position range {pos_range} not implemented.")

    def generate_init_env(self) -> None:
        """Generate an initial environment with special states."""
        self._reset_grid()

        # Place goal
        # Skip if goal is null
        if self.args.goal_num:
            self._place_constrained_state(
                "goal", self.args.goal_num, self.args.goal_range
            )

        # Place death states
        if self.args.death_num:
            self._place_constrained_state("death", self.args.death_num, "random")

        # Place ice states
        if self.args.ice_area_num:
            for _ in range(self.args.ice_area_num):
                max_ice_size = self.args.ice_max_size
                ice_width = random.randint(1, max_ice_size)
                ice_height = random.randint(1, max_ice_size)
                self._place_square_area("ice", ice_width, ice_height)

        # Place obstacles
        if (
            self.args.obstacle_line_num
        ):  # TODO: Constrain it to be not at the border line
            for _ in range(self.args.obstacle_line_num):
                max_obstacle_size = self.args.obstacle_max_size
                obstacle_num = random.randint(1, max_obstacle_size)
                if np.random.rand() < 0.5:
                    self._place_square_area("obstacle", 1, obstacle_num)
                else:
                    self._place_square_area("obstacle", obstacle_num, 1)

        # Place portals (in pairs)
        if self.args.portal_num:
            self._place_constrained_state("portal", self.args.portal_num, "border")

    def create_one_grid(self) -> GridWorld:
        # Create a more complex environment with obstacles, ice states, death states, and portals
        obstacles = {
            (x, y): StateType.OBSTACLE for (x, y) in self.state_positions["obstacle"]
        }
        ice_states = {(x, y): StateType.ICE for (x, y) in self.state_positions["ice"]}
        death_states = {
            (x, y): StateType.DEATH for (x, y) in self.state_positions["death"]
        }
        portals = {
            (x, y): StateType.PORTAL for (x, y) in self.state_positions["portal"]
        }
        goals = {(x, y): StateType.GOAL for (x, y) in self.state_positions["goal"]}

        full_env = GridWorld(
            self.width,
            self.height,
            state_types={**obstacles, **ice_states, **death_states, **portals, **goals},
        )

        return full_env

    def generate_followup_env(self, last_env=None) -> None:
        """Generate a follow-up environment based on the last environment."""
        # TODO: Implement the generation of follow-up environments -> based on curriculum option, could be the same as the last one or depends on the complexity measure
        pass

    def visualize_grid(self, save_path=None, iter=0):
        """Visualize the grid environment with special states."""

        # Set up the plot
        # TODO: Title should be the name of the special states configuration
        # Create a blank plot
        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw the grid lines
        grid_width, grid_height = self.width, self.height
        for x in range(grid_width + 1):
            ax.plot([x, x], [0, grid_height], color="black", linewidth=0.5)
        for y in range(grid_height + 1):
            ax.plot([0, grid_width], [y, y], color="black", linewidth=0.5)

        # Add icons for different states
        if "obstacle" in self.state_positions:
            for x, y in self.state_positions["obstacle"]:
                rect = plt.Rectangle((x, y), 1, 1, facecolor="gray", edgecolor="black")
                ax.add_patch(rect)
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    "#",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=20,
                )

        if "ice" in self.state_positions:
            for x, y in self.state_positions["ice"]:
                rect = plt.Rectangle(
                    (x, y), 1, 1, facecolor="#89CFF0", edgecolor="black"
                )
                ax.add_patch(rect)
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    "❄️",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=20,
                )

        if "death" in self.state_positions:
            for x, y in self.state_positions["death"]:
                rect = plt.Rectangle((x, y), 1, 1, facecolor="black", edgecolor="black")
                ax.add_patch(rect)
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    "☠️",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=20,
                )

        if "portal" in self.state_positions:
            for x, y in self.state_positions["portal"]:
                rect = plt.Rectangle(
                    (x, y), 1, 1, facecolor="#AFE1AF", edgecolor="black"
                )
                ax.add_patch(rect)
                plt.text(
                    x + 0.5,
                    y + 0.5,
                    "✈️",
                    color="white",
                    ha="center",
                    va="center",
                    fontsize=20,
                )
        for x, y in self.state_positions["goal"]:
            rect = plt.Rectangle((x, y), 1, 1, facecolor="#ebc934", edgecolor="black")
            ax.add_patch(rect)
            plt.text(
                x + 0.5,
                y + 0.5,
                "★",
                color="white",
                ha="center",
                va="center",
                fontsize=20,
            )

        # Set aspect of the plot to be equal
        ax.set_aspect("equal")

        # Set limits and labels
        ax.set_xlim(0, grid_width)
        ax.set_ylim(0, grid_height)
        ax.set_xticks(range(grid_width + 1))
        ax.set_yticks(range(grid_height + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Save the figure
        if save_path:
            plt.savefig(f"{save_path}/grid_{iter}.png")
            plt.close(fig)

    # ----- Complexity Measurement Functions -----
    def compute_state_distribution_complexity(self) -> float:
        """Counts each state type and computes a weighted complexity."""
        weights = {"death": 4, "ice": 3, "obstacle": 2, "portal": 1, "goal": 5}
        complexity = sum(
            weights[state] * len(self.state_positions[state]) for state in weights
        )
        return complexity

    def compute_path_complexity(self) -> float:
        """Estimates path complexity based on distance from a random start to the goal."""
        if not self.state_positions["goal"]:
            return 0

        start = (0, 0)  # TODO: Randomize starting position in the class
        goal = self.state_positions["goal"][0]
        path_length = self._shortest_path_length(start, goal)
        return path_length if path_length is not None else float("inf")

    def compute_proximity_complexity(self):
        """Computes complexity based on the average proximity of special states."""
        all_positions = []
        for state in self.state_positions:
            all_positions.extend(self.state_positions[state])

        if len(all_positions) <= 1:
            return 0

        # Compute pairwise distances
        distances = cdist(all_positions, all_positions, metric="euclidean")
        mean_distance = np.mean(distances)
        return 1 / mean_distance if mean_distance > 0 else float("inf")

    def compute_stochasticity_complexity(self):
        """Returns a weighted score based on the number of stochastic states (ice)."""
        return len(self.state_positions["ice"]) * 2  # Weight of 2 for ice

    def compute_obstacle_density(self):
        """Returns the proportion of obstacles in the grid."""
        total_cells = self.width * self.height
        obstacle_count = len(self.state_positions["obstacle"])
        return obstacle_count / total_cells

    def compute_total_complexity(self) -> float:
        """Combines all complexity measures into a single score."""
        # Complexity measures regarding path from start to goal
        C_path = self.compute_path_complexity()

        # Summary statistics for state distribution
        # regarding state distribution with the goal
        C_proximity = self.compute_proximity_complexity()
        # regarding state amount
        C_state = self.compute_state_distribution_complexity()

        # Complexity measures regarding different states
        C_stochasticity = self.compute_stochasticity_complexity()
        C_obstacle = self.compute_obstacle_density()

        # Combine with weights
        total_complexity = (
            0.4 * C_state
            + 0.3 * C_path
            + 0.2 * C_proximity
            + 0.1 * C_stochasticity
            + 0.2 * C_obstacle
        )
        return total_complexity

    # Helper function for path complexity
    def _shortest_path_length(self, start, goal):
        """Computes shortest path length using A*."""
        # TODO: should penalize the cells in the path which are special states

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Priority queue for A*
        pq = PriorityQueue()
        pq.put((0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while not pq.empty():
            _, current = pq.get()

            if current == goal:
                return cost_so_far[current]

            x, y = current
            neighbors = [
                (x + dx, y + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height
            ]

            for next_cell in neighbors:
                if self.grid[next_cell[1], next_cell[0]] == "O":  # Obstacle
                    continue
                new_cost = cost_so_far[current] + 1
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + heuristic(next_cell, goal)
                    pq.put((priority, next_cell))
                    came_from[next_cell] = current

        return None  # No path found

    # ----- Complexity Measurement Functions -----


class GridEnvTransition(EnvTransition):
    def __init__(self, width, height, state_types=None, random_seed=None):
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4


class PartiallyObservableGridEnvTransition(GridEnvTransition):
    def __init__(self, width, height, state_types=None, random_seed=None):
        self.width = width
        self.height = height
        self.num_states = width * height
        self.num_actions = 4
