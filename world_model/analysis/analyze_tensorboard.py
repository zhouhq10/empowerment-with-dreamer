import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

# from skimage.color import rgb2gray
from scipy.stats import entropy

EPS = 1e-10

# Path to the replay folder
env_names = [
    "extrinsic",
    "obsnovelty",
    "obscbet",
    "latentsnovelty",
    "latentsinfogain",
    "latentsempowerment",
]
base_path = "/mnt/qb/work/wu/wkn601/dreamer_pretrain/wall/SimpleCrossingS9N1/{env_name}"
replay_folder = f"{base_path}/replay"

# Key: image, Shape: (1024, 64, 64, 3)
# Key: direction, Shape: (1024,)
# Key: reward, Shape: (1024,)
# Key: intrinsic_reward, Shape: (1024,)
# Key: is_first, Shape: (1024,)
# Key: is_last, Shape: (1024,)
# Key: is_terminal, Shape: (1024,)
# Key: extrinsic_reward, Shape: (1024,)
# Key: intrinsic_reward_obs_novelty, Shape: (1024,)
# Key: action, Shape: (1024, 7)
# Key: reset, Shape: (1024,)
# Key: id, Shape: (1024, 16)


def visualize_image(img):
    # Temporary
    plt.imshow(img)
    plt.savefig("minigrid_image.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def calculate_sequence_entropy(image_sequence):
    # Combine all pixel values from the entire sequence
    all_pixels = []
    img_num = image_sequence.shape[0]
    for index in range(img_num):
        image = image_sequence[index]
        all_pixels.append(image.reshape(-1))

    all_pixels = np.array(all_pixels)

    overall_entropy = 0
    pixel_num = all_pixels.shape[-1]
    for i in range(pixel_num):
        # Calculate the histogram
        histogram, _ = np.histogram(all_pixels[:, i], bins=256, density=True)
        # Calculate entropy
        overall_entropy += entropy(histogram, base=2)

    return overall_entropy / pixel_num


def calculate_action_entropy(actions):
    one_hot_array = (actions == actions.max(axis=1, keepdims=True)).astype(int)

    # Step 1: Compute category probabilities
    category_totals = np.sum(one_hot_array, axis=0)
    probabilities = category_totals / np.sum(category_totals)

    # Step 2: Compute entropy
    # Add a small epsilon to avoid log(0)
    entropy = -np.sum(probabilities * np.log(probabilities + EPS))

    return entropy


def analyze_episodes(episodes):
    mean_score_across_episodes = []
    mean_length_across_episodes = []
    mean_intrinsic_rewards = []
    mean_extrinsic_rewards = []
    obs_entropy = []
    acts_entropy = []

    for episode in episodes:
        obs_entropy.append(calculate_sequence_entropy(episode["image"]))
        acts_entropy.append(calculate_action_entropy(episode["action"]))
        mean_score_across_episodes.append(episode["reward"].sum())
        mean_length_across_episodes.append(len(episode["reward"]) - 1)
        mean_intrinsic_rewards.append(episode["intrinsic_reward"].mean())
        mean_extrinsic_rewards.append(episode["extrinsic_reward"].mean())

    return {
        "mean_score_across_episodes": mean_score_across_episodes,
        "mean_length_across_episodes": mean_length_across_episodes,
        "mean_intrinsic_rewards": mean_intrinsic_rewards,
        "mean_extrinsic_rewards": mean_extrinsic_rewards,
        "obs_entropy": obs_entropy,
        "acts_entropy": acts_entropy,
    }


def get_episodes(replay_folder):
    episodes = []
    old_is_first = None
    old_episode = None
    # Iterate over all .npz files in the replay folder
    for filename in os.listdir(replay_folder)[:10]:
        if filename.endswith(".npz"):
            filepath = os.path.join(replay_folder, filename)

            # Extract every episode
            data = np.load(filepath, allow_pickle=True)
            features = list(data.keys())
            # for key, value in data.items():
            #     print(f"Key: {key}, Shape: {value.shape}")

            # Extract relevant keys
            is_first_index = np.where(data["is_first"] == 1)[0]
            is_last_index = np.where(data["is_last"] == 1)[0]
            is_terminal_index = np.where(data["is_terminal"] == 1)[0]

            for i in range(len(is_first_index)):
                start_index = int(is_first_index[i])

                if i == 0 and start_index > int(
                    min(is_last_index[i], is_terminal_index[i])
                ):
                    assert old_episode is not None
                    episode = {key: value[:start_index] for key, value in data.items()}
                    episode = {
                        key: np.concatenate([old_episode[key], episode[key]], axis=0)
                        for key in episode.keys()
                    }
                    episodes.append(episode)
                    continue

                try:
                    end_index = int(min(is_last_index[i], is_terminal_index[i]))
                    assert start_index < end_index
                    episodes.append(
                        {
                            key: value[start_index:end_index]
                            for key, value in data.items()
                        }
                    )
                except:
                    old_is_first = int(is_first_index[-1])
                    old_episode = {
                        key: value[old_is_first:] for key, value in data.items()
                    }
    return episodes


def plot_results(all_env_results):
    subplot_features = [
        "mean_score_across_episodes",
        "mean_length_across_episodes",
        "mean_intrinsic_rewards",
        "mean_extrinsic_rewards",
        "obs_entropy",
        "acts_entropy",
    ]
    # Plot individual figure for each one and save it
    for i, feature in enumerate(subplot_features):
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        for env_name, results in all_env_results.items():
            axs.plot(results[feature], label=env_name)
        axs.set_title(feature)
        axs.legend()
        fig.savefig(f"{feature}.png", bbox_inches="tight", pad_inches=0)
        import ipdb

        ipdb.set_trace()


def main():
    all_env_results = {}
    for env_name in env_names:
        replay_folder = f"{base_path.format(env_name=env_name)}/replay"
        episodes = get_episodes(replay_folder)
        results = analyze_episodes(episodes)
        all_env_results[env_name] = results

    plot_results(all_env_results)


if __name__ == "__main__":
    main()


# # Path to the directory containing the .tfevents file
# logdir = "/mnt/qb/work/wu/wkn601/dreamer_pretrain/wall/SimpleCrossingS9N1/obsnovelty"

# # Locate the .tfevents file
# event_files = [f for f in os.listdir(logdir) if f.startswith("events.out.tfevents")]
# if not event_files:
#     raise FileNotFoundError("No .tfevents file found in the directory.")

# # Initialize a DataFrame to store extracted data
# data = []

# import ipdb

# ipdb.set_trace()
# # Extract data from the .tfevents file
# for event_file in event_files:
#     event_path = os.path.join(logdir, event_file)
#     for event in tf.compat.v1.train.summary_iterator(event_path):
#         for value in event.summary.value:
#             # Store each metric in a structured way
#             data.append(
#                 {"step": event.step, "tag": value.tag, "value": value.simple_value}
#             )

# import ipdb

# ipdb.set_trace()
# # Convert to a pandas DataFrame for analysis
# df = pd.DataFrame(data)
# print(df.head())

# # Save the extracted data to a CSV file for further analysis
# df.to_csv("tfevents_extracted.csv", index=False)
