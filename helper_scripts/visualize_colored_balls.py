"""
Render the MixedEnv with Ball objects in two colour variants:
  - grey   (same colour as walls)
  - orange (same hue as lava)

This script is self-contained and does NOT import environment.py, so it avoids
the scipy/numpy version conflict present in some conda environments.

Usage (from src/ or repo root, in any env that has minigrid + matplotlib):
    python src/visualize_colored_balls.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import WorldObj, Ball, Lava, Wall, Goal
from minigrid.utils.rendering import fill_coords, point_in_rect
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT, COLORS

# ── Register 'orange' in MiniGrid's colour palette ────────────────────────────
# Lava is drawn via a custom fill rather than COLORS, so there is no built-in
# 'orange' entry.  We add one here that matches lava's visual hue.
if "orange" not in COLORS:
    COLORS["orange"] = np.array([255, 128, 0])

# ── Minimal Ice tile (copied from environment.py, no scipy dependency) ─────────
if "ice" not in OBJECT_TO_IDX:
    OBJECT_TO_IDX["ice"] = len(OBJECT_TO_IDX)
    IDX_TO_OBJECT[OBJECT_TO_IDX["ice"]] = "ice"


class Ice(WorldObj):
    def __init__(self):
        super().__init__("ice", "blue")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


# # ── MixedEnv layout (grid only, no stochastic step logic needed for rendering) ─

# ICE_POSITIONS = [
#     (1, 1), (1, 6), (3, 6), (4, 4), (4, 5), (5, 5), (4, 6), (5, 6),
#     (4, 7), (4, 8), (6, 4), (6, 5), (8, 9), (5, 1), (8, 2), (8, 3),
#     (9, 2), (9, 3),
# ]
# LAVA_POSITIONS = [
#     (2, 10), (3, 7), (5, 4), (9, 7), (4, 1), (5, 2), (6, 6), (1, 8),
# ]
# WALL_POSITIONS = [
#     (7, 4), (7, 3), (7, 2), (7, 5), (8, 5), (9, 5), (10, 5),
# ]
ICE_POSITIONS = []
LAVA_POSITIONS = []
WALL_POSITIONS = []
# Cells that are empty in the default layout — adjust to your needs
BALL_POSITIONS = [
    (2,2), (3,5), (5, 6), (7, 4), (6, 3), 
]


class MixedEnvWithBalls(MiniGridEnv):
    """MixedEnv layout with configurable-colour Ball objects."""

    def __init__(self, ball_color: str = "blue", size: int = 12, **kwargs):
        self._ball_color = ball_color
        mission_space = MissionSpace(mission_func=lambda: "grand mission")
        super().__init__(
            mission_space=mission_space,
            # grid_size=(5,7),
            width=9,
            height=7,
            max_steps=10000,
            see_through_walls=False,
            highlight=False,      # disable the field-of-view triangle overlay
            **kwargs,
        )
        # Fixed start position matching original MixedEnv
        self.agent_start_pos = (1,1)
        self.agent_start_dir = 0

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for x, y in ICE_POSITIONS:
            self.put_obj(Ice(), x, y)
        for x, y in LAVA_POSITIONS:
            self.put_obj(Lava(), x, y)
        for x, y in WALL_POSITIONS:
            self.put_obj(Wall(), x, y)

        self.put_obj(Goal(), width - 2, height - 2)

        # Place balls only in empty cells
        for x, y in BALL_POSITIONS:
            if self.grid.get(x, y) is None:
                self.put_obj(Ball(color=self._ball_color), x, y)

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = "grand mission"


# ── Render helper ──────────────────────────────────────────────────────────────

def render(ball_color: str) -> np.ndarray:
    env = MixedEnvWithBalls(ball_color=ball_color, render_mode="rgb_array")
    env.reset()
    img = env.render()
    env.close()
    return img


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    img_grey   = render("grey")
    img_orange = render("orange")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(img_grey)
    axes[0].set_title("Grey balls  (wall colour)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(img_orange)
    axes[1].set_title("Orange balls  (lava colour)", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "colored_balls_comparison.pdf")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {os.path.normpath(out_path)}")
    plt.show()
