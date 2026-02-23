import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

from multi_plot import plot_scores

# window = 2e5
window = 1
transfer_step_limit = crafter_step_limit = 1e6
tabula_minigrid_step_limit = 1e5

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.subplots_adjust(wspace=0.1)

base_path = "/mnt/qb/work/wu/wkn601/dreamer_transfer"
envs = ["wall", "lava", "dynamics"]
rewards = [
    # "extrinsic",
    # "obsnovelty",
    "latentsnovelty_32",
    "latentsinfogain",
    "latentsempowerment",
]

minigrid_results = {
    "wall": f"{base_path}/wall/SimpleCrossingS9N1_SimpleCrossingS9N2",
    "lava": f"{base_path}/lava/DistShift1_DistShift2",
    "dynamics": f"{base_path}/dynamics/Dynamic-Obstacles-16x16_Dynamic-Obstacles-8x8",
}


def process_and_average_scores(
    base_path,
    filename,
    process_func,
    score_key,
    step_limit=None,
    num_eval_episodes=None,
    window=200000,
):
    filepath = f"{base_path}/{filename}"
    scores = (
        process_func(filepath, score_key, step_limit, num_eval_episodes)
        if score_key
        else process_func(filepath, step_limit, num_eval_episodes)
    )
    return average_scores_within_window(scores, window, step_limit)


def average_scores_within_window(scores, window=200000, step_limit=1e6):
    scores.sort(key=lambda x: x[0])  # sort by step
    averaged_scores = []

    # Dummy value
    averaged_scores.append((0, 0, 0))
    index = 0

    # Set step limit to min of the last step
    step_limit = min(step_limit, scores[-1][0])

    for step_width in range(0, int(step_limit), int(window)):
        step_sum = 0
        mean_sum = 0
        count = 0

        while index < len(scores) and scores[index][0] < step_width + window:
            step, mean, _ = scores[index]
            step_sum += step
            mean_sum += mean
            count += 1
            index += 1

        # Calculate the std using mean
        std = np.std(
            [mean for s, mean, _ in scores if step_width <= s < step_width + window]
        )

        if count > 0:
            averaged_scores.append((step_width + window, mean_sum / count, std))

    return averaged_scores


def process_dreamer_scores(filepath, return_col, step_limit, num_eval_episodes):

    with open(filepath, "r") as file:
        data = [json.loads(line) for line in file]
    # Go over all rows and extract the return_col field and step if they exist
    scores = []

    for row in data:
        if return_col in row and "step" in row:
            if row["step"] > step_limit:
                break
            score = float(row[return_col])
            step = int(row["step"])
            scores.append((step, score))

    # For all elements which have the same step, find mean and std
    scores.sort(key=lambda x: x[0])
    scores = [
        (
            step,
            np.mean([score for s, score in scores if s == step]),
            np.std([score for s, score in scores if s == step]),
        )
        for step, _ in scores
    ]

    # Divide std by sqrt of number of elements
    scores = [
        (step, mean, std / np.sqrt(num_eval_episodes)) for step, mean, std in scores
    ]

    # remove duplicates and sort
    scores = list(set(scores))
    scores.sort(key=lambda x: x[0])

    return scores


def average_between_experiments(scores, window, step_limit, num_experiments):

    if num_experiments == 1:
        return scores[0]

    mean_scores = []

    # Data of the form [(step, mean, std), ...] for each experiment
    # Iterate over the scores, calculate mean using individual means and std from these. Step is the same
    for i in range(len(scores[0])):
        step = scores[0][i][0]
        mean = np.mean([scores[j][i][1] for j in range(num_experiments)])
        std = np.std([scores[j][i][1] for j in range(num_experiments)])
        mean_scores.append((step, mean, std / np.sqrt(num_experiments)))

    return mean_scores


def get_dataframe(
    path,
    num_eval_episodes=8,
    step_limit=1e6,
    window=200000,
    mode="extrinsic",
    num_experiments=1,
):
    dreamer_filename = "scores.jsonl"
    dreamer_score_key = "eval_episode/score"
    dreamer_intrinsic_score_key = "episode/intrinsic_return"

    # Create empty object to store scores among all experiments
    scores = []

    for i in range(0, num_experiments):
        score = process_and_average_scores(
            base_path=path.replace("#", str(i+1)) if "#" in path else path,
            filename=dreamer_filename,
            process_func=(
                process_dreamer_scores
                if mode == "extrinsic"
                else process_intrinsic_dreamer_scores
            ),
            score_key=(
                dreamer_score_key
                if mode == "extrinsic"
                else dreamer_intrinsic_score_key
            ),
            step_limit=step_limit,
            num_eval_episodes=num_eval_episodes,
            window=window,
        )

        scores.append(score)

    return average_between_experiments(scores, window, step_limit, num_experiments)


def process_intrinsic_dreamer_scores(
    filename, return_col, step_limit, num_eval_episodes
):
    import ipdb

    ipdb.set_trace()

    with open(filename, "r") as file:
        data = [json.loads(line) for line in file]

    scores = []
    for row in data:
        if return_col in row and "step" in row:
            if row["step"] > step_limit:
                break

            score = float(row[return_col])
            step = int(row["step"])
            scores.append((step, score))

    # For all elements which have the same step, find mean and std
    scores.sort(key=lambda x: x[0])
    scores = [
        (
            step,
            np.mean([score for s, score in scores if s == step]),
            np.std([score for s, score in scores if s == step]),
        )
        for step, _ in scores
    ]

    # remove duplicates and sort
    scores = list(set(scores))
    scores.sort(key=lambda x: x[0])

    return scores


def create_df(data, label):
    df = pd.DataFrame(data, columns=["step", "mean_return", "std_error_return"])
    df["label"] = label
    return df

import matplotlib.pyplot as plt

def plot_dynamics_scores(all_data, env, step_limit=1e6, hide_y_ticks=True):

    data = all_data[env]
    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Colors for the different keys
    colors = plt.cm.tab10.colors

    for i, (key, df) in enumerate(data.items()):
        steps = df["step"].to_numpy()
        mean_returns = df["mean_return"].to_numpy()
        std_errors = df["std_error_return"].to_numpy()
        
        plt.plot(steps, mean_returns, label=key, color=colors[i % len(colors)])
        plt.fill_between(steps, 
                        mean_returns - std_errors,
                        mean_returns + std_errors,
                        color=colors[i % len(colors)], alpha=0.2)

    # Adding labels, title, legend, and grid
    plt.xlabel("Steps")
    plt.ylabel("Mean Return")
    plt.title("Visualization of Return in {}".format(env))
    plt.legend(title="Keys")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{env}_new.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()

def plot_bar_scores(all_data, env, step_limit=1e6, hide_y_ticks=True):
    # Draw the bar plot of the scores
    plt.figure(figsize=(10, 6))
    data = all_data[env]
    # Colors for the different keys
    colors = plt.cm.tab10.colors

    for i, (key, df) in enumerate(data.items()):
        steps = df["step"].to_numpy()
        mean_returns = df["mean_return"].to_numpy()
        std_errors = df["std_error_return"].to_numpy()
        
        plt.bar(key, mean_returns.mean(), yerr=mean_returns.std()/np.sqrt(len(mean_returns)), color=colors[i % len(colors)], label=key)
    
    # Adding labels, title, legend, and grid
    plt.xlabel("Steps")
    plt.ylabel("Mean Return")
    plt.title("Visualization of Return in {}".format(env))
    plt.legend(title="Keys")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{env}_bar.png", dpi=300, bbox_inches="tight")

    # Show the plot
    plt.show()


def main():
    # Generate all and then concatenate
    dfs = {}
    for name, path in minigrid_results.items():
        dfs[name] = {}
        for reward in rewards:
            cur_path = f"{path}/exp#/{reward}"
            # cur_path = f"{path}/{reward}"

            scores = get_dataframe(
                cur_path,
                num_eval_episodes=8,
                step_limit=tabula_minigrid_step_limit,
                window=window,
                mode="extrinsic",
                num_experiments=2,
            )
            df = create_df(scores, f'{name}_{reward}')
            dfs[name][reward] = df
    
    # Plotting
    for env, path in minigrid_results.items():
        plot_dynamics_scores(dfs, env)
        plot_bar_scores(dfs, env)



if __name__ == "__main__":
    main()


# handles, labels = plot_scores(df, 'Extrinsic', 'Tabula Rasa MiniGrid', axs[0, 0], y_lim=1, step_limit=tabula_minigrid_step_limit)

# transfer_minigrid_results = {
#     'DreamerV3 (CBET)': 'dreamerv3/minigrid-transfer-1M-Unlock/',
#     'IMPALA (CBET)': 'impala/minigrid-transfer-1M-unlock/'
# }

# df = generate_dfs(transfer_minigrid_results, window=window, step_limit=transfer_step_limit, num_experiments=1)
# plot_scores(df, 'Extrinsic', 'Transfer MiniGrid', axs[0, 1], y_lim=1, hide_y_ticks=False)
