"""CLI entry point for running a single agent experiment.

Usage example::

    python run_single_agent.py \\
        --agent ps \\
        --rewards info_gain,empowerment \\
        --combination mean \\
        --gamma 0.9 \\
        --lr 1.0 \\
        --q-init 0.0 \\
        --seed 42 \\
        --n-steps 10000 \\
        --eval-interval 100

Results are saved to disk via :func:`run_utils.run_or_load` and re-loaded
on subsequent calls with the same configuration.
"""

import sys
import os
import argparse
from environment import MixedEnv, AgentPosAndDirWrapper, get_all_states
from agent import PrioritizedSweepingAgent
from run_utils import run_or_load


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description='Run single agent experiment')
    parser.add_argument('--agent', type=str, required=True, choices=['ps'],
                        help='Type of agent to run')
    parser.add_argument('--rewards', type=str, default='none',
                        help='Comma-separated list of rewards (info_gain,empowerment)')
    parser.add_argument('--combination', type=str, choices=['mean', 'product'],
                        help='How to combine multiple rewards')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--q-init', type=float, help='Initial Q-value')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--n-steps', type=int, default=10000, help='Number of steps')
    parser.add_argument('--eval-interval', type=int, default=100, help='Evaluation interval')

    args = parser.parse_args()

    # Validate that --combination is provided iff there are multiple rewards
    rewards = args.rewards.split(',') if args.rewards != 'none' else []
    if len(rewards) > 1 and not args.combination:
        parser.error("--combination is required when using multiple rewards")
    if len(rewards) <= 1 and args.combination:
        parser.error("--combination should only be used with multiple rewards")
    return args


def main() -> None:
    NUM_ACTIONS = 3  # left, right, forward

    args = parse_args()

    # Ensure imports work regardless of the caller's working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Build and reset the environment (reset is required so get_all_states
    # does not incorrectly include wall positions)
    env = MixedEnv()
    env = AgentPosAndDirWrapper(env)
    env.reset()

    if args.agent == "random":
        raise NotImplementedError("Random agent not implemented anymore")
    else:
        rewards = args.rewards.split(',') if args.rewards != 'none' else []

        # Default reward computation configs forwarded to each reward function
        DEFAULT_REWARD_CFG = {
            "empowerment": {
                "num_steps": 1,
                "method": "blahut_arimoto",
            },
            "info_gain": {
                "method": "LittleSommerPIG"
            },
            "novelty": {}
        }

        if args.agent == "q":
            raise NotImplementedError("Q-learning agent not implemented anymore")
        else:
            model_kwargs = {
                "reward_types": rewards,
                "reward_configs": DEFAULT_REWARD_CFG,
                "combination_method": args.combination
            }
            agent = PrioritizedSweepingAgent(
                num_actions=NUM_ACTIONS,
                all_states=get_all_states(env.unwrapped),
                learning_rate=args.lr,
                gamma=args.gamma,
                q_init=args.q_init,
                model_kwargs=model_kwargs
            )

    history = run_or_load(env, agent, rewards, args.n_steps, args.eval_interval,
                          combination_method=args.combination, seed=args.seed,
                          custom_identifier="final")


if __name__ == "__main__":
    main()
