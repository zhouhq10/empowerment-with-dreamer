import os

GAMMA = [0.999] # NOTE: USE FLOAT VALUES ONLY
LR = [1.0] # NOTE: USE FLOAT VALUES ONLY
Q_INIT = [0.0] # NOTE: USE FLOAT VALUES ONLY
LEARNING_AGENTS = ["ps"]
REWARDS = ["info_gain", "empowerment", "info_gain,empowerment", "none", "novelty"]
# REWARDS = ["novelty"]
# REWARDS = ["info_gain,empowerment"]
REWARD_COMBINATIONS = ["product", "mean"]
RANDOM_SEEDS = [36743274, 28958, 63, 8888888, 412547]
#RANDOM_SEEDS = [8888888, 412547]

template_learning = """#!/bin/bash
#SBATCH -J {agent_type}_g{gamma}_l{lr}_q{q_init}_{reward_string}{combination_str}_seed{seed}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=2080-galvani
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00
#SBATCH --mem=10G
#SBATCH --output=/mnt/lustre/work/wu/wkn758/{agent_type}_g{gamma}_l{lr}_q{q_init}_{reward_string}{combination_str}_seed{seed}-%j.out
#SBATCH --error=/mnt/lustre/work/wu/wkn758/{agent_type}_g{gamma}_l{lr}_q{q_init}_{reward_string}{combination_str}_seed{seed}-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=freddy.mantiuk@gmail.com

source $HOME/.bashrc
conda activate /mnt/lustre/work/wu/wkn758/conda_envs/empower

srun python3 {src_path}/run_single_agent.py \\
    --agent {agent_type} \\
    --rewards {rewards} \\
    {combination_arg}--gamma {gamma} \\
    --lr {lr} \\
    --q-init {q_init} \\
    --seed {seed}

conda deactivate
"""

template_random = """#!/bin/bash
#SBATCH -J random_agent_seed{seed}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=2080-galvani
#SBATCH --time=0-03:00
#SBATCH --mem=10G
#SBATCH --output=/mnt/lustre/work/wu/wkn758/random_agent_seed{seed}-%j.out
#SBATCH --error=/mnt/lustre/work/wu/wkn758/random_agent_seed{seed}-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=freddy.mantiuk@gmail.com

source $HOME/.bashrc
conda activate /mnt/lustre/work/wu/wkn758/conda_envs/empower

srun python3 {src_path}/run_single_agent.py \
    --agent random \
    --seed {seed}

conda deactivate
"""

# Change working directory to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
src_path = os.getcwd()

# make sure that all seeds are between 0 and 2^32 - 1
for seed in RANDOM_SEEDS:
    assert 0 <= seed < 2**32 - 1, f"Seed {seed} is not in the valid range [0, 2^32 - 1]"

# Create directory for job scripts
os.makedirs("../jobs", exist_ok=True)

# Generate random agent jobs --> replaced with "none" reward
#for seed in RANDOM_SEEDS:
#    with open(f"../jobs/random_agent_seed{seed}.sh", "w") as f:
#        f.write(template_random.format(src_path=src_path, seed=seed))

# Generate learning agent jobs
for rewards in REWARDS:
    for agent_type in LEARNING_AGENTS:
        for gamma in GAMMA:
            for lr in LR:
                for q_init in Q_INIT:
                    for seed in RANDOM_SEEDS:
                        for combination in REWARD_COMBINATIONS:
                            combination_str = f"_{combination}" if len(rewards.split(",")) > 1 else ""
                            combination_arg = f"--combination {combination} " if len(rewards.split(",")) > 1 else ""
                            reward_str = "_".join(rewards.split(","))

                            script = template_learning.format(
                                src_path=src_path,
                                agent_type=agent_type,
                                rewards=rewards,
                                reward_string=reward_str,
                                combination_str=combination_str,
                                combination_arg=combination_arg,
                                gamma=float(gamma),
                                lr=float(lr),
                                q_init=float(q_init),
                                seed=seed
                            )
                                                        
                            filename = f"../jobs/{agent_type}_g{float(gamma)}_l{float(lr)}_q{float(q_init)}_{reward_str}{combination_str}_seed{seed}.sh"
                            with open(filename, "w") as f:
                                f.write(script)