import sys

sys.path.append("..")

from env_generation.envs.grid import GridWorld, StateType
from env_generation.intrinsics.empowerment import Empowerment
from env_generation.envs.env_transition import EnvTransition
import argparse


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)

    # Setting a series environments
    parser.add_argument(
        "--env_series",
        type=str,
        help="The series of environments to generate, e.g. 'same', 'increasing_difficulty', 'increasing_state'",
    )
    parser.add_argument(
        "--env_series_num",
        type=int,
    )
    parser.add_argument(
        "--env_num_per_series",
        type=int,
    )

    # Setting states for single environments - this setting is constrained for the first environment in a series
    parser.add_argument("--goal_num", type=int, default=1)
    parser.add_argument("--goal_range", type=str, default="last_column")

    parser.add_argument("--death_num", type=int, default=1)

    parser.add_argument("--ice_max_size", type=int, default=2)
    parser.add_argument("--ice_area_num", type=int, default=2)

    parser.add_argument("--portal_num", type=int, default=1)

    parser.add_argument("--obstacle_max_size", type=int, default=4)
    parser.add_argument("--obstacle_line_num", type=int, default=2)

    args = parser.parse_args()

    env_transit = EnvTransition(args.width, args.height, args)
    env_transit.generate_init_env()
    # grid = env_transit.create_one_grid()
    # env_transit.visualize_grid(save_path="..")
    complexity = env_transit.compute_total_complexity()


if __name__ == "__main__":
    main()
