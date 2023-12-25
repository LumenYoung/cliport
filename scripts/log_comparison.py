import os
from pathlib import Path
import orjson
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def longest_common_prefix(strs: List[str]):
    if not strs:
        return ""
    shortest_str = min(strs, key=len)
    for i, char in enumerate(shortest_str):
        for other in strs:
            if other[i] != char:
                return shortest_str[:i]
    return shortest_str


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), "valid") / window_size


def load_json_log(filepath: Path) -> Dict:
    if not filepath.exists():
        raise ValueError(f"No such file: {filepath}")

    with open(filepath, "rb") as f:
        data = orjson.loads(f.read())

    return data


def extract_step_rewards(data: Dict) -> List[float]:
    result = []

    for key, value in data.items():
        if not key.startswith("epoch_"):
            continue

        for step_key, step_value in value.items():
            if not step_key.startswith("step_"):
                continue
            result.append(float(step_value["reward"]))

    return result


def extract_step_success(data: Dict) -> List[bool]:
    result = []

    for key, value in data.items():
        if not key.startswith("epoch_"):
            continue

        for step_key, step_value in value.items():
            if not step_key.startswith("step_"):
                continue
            result.append(step_value["done"])

    return result


def extract_grouped_step_rewards_and_successes(
    data: Dict,
) -> Tuple[List[List[float]], List[List[bool]]]:
    rewards = []

    successes = []

    for key, value in data.items():
        if not key.startswith("epoch_"):
            continue
        curr_epoch_reward = []
        curr_epoch_success = []
        for step_key, step_value in value.items():
            if not step_key.startswith("step_"):
                continue
            curr_epoch_reward.append(float(step_value["reward"]))
            curr_epoch_success.append(step_value["done"])

        rewards.append(curr_epoch_reward)
        successes.append(curr_epoch_success)

    return rewards, successes


def plot_datas(
    datas: List[np.array],
    file_name: str,
    xlabel: str = "Epochs",
    ylabel: str = "Reward",
    smooth_window_size: Optional[int] = None,
) -> None:
    plt.figure(figsize=(10, 6))  # Set the figure size

    for data in datas:
        if smooth_window_size is not None:
            data = moving_average(data, smooth_window_size)
        plt.plot(data, label=filename)  # Plot the rewards

    plt.xlabel(xlabel)  # Set the x-axis label
    plt.ylabel(ylabel)  # Set the y-axis label

    dirname = os.path.dirname(file_name)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    plt.savefig(file_name)


if __name__ == "__main__":
    log_dir = "./logs"
    plot_dir = "./plots"

    log_files = os.listdir(log_dir)

    log_files = [
        "palletizing-boxes-100_demos-2023-12-16.json",
        "palletizing-boxes-correction_5examples-100_demos-2023-12-16.json",
        "palletizing-boxes-correction_5examples-correction_feedback-100_demos-2023-12-18.json",
        # "towers-of-hanoi-seq-full-100_demos-2023-12-17.json",
        # "towers-of-hanoi-seq-full-correction_5examples-100_demos-2023-12-18.json",
        # "block-insertion-correction_5examples-correction_feedback-100_demos-2023-12-19.json",
        # "block-insertion-100_demos-2023-12-16.json",
    ]

    plt.figure(figsize=(10, 6))  # Set the figure size

    windowsize = 10

    comp_rewards = []
    comp_successes = []

    prefix = longest_common_prefix(log_files)

    overall_steps = 0

    for filename in log_files:
        data = load_json_log(Path(log_dir, filename))

        rewards, successes = extract_grouped_step_rewards_and_successes(data)

        steps = [len(re) for re in rewards]
        overall_rewards = [np.array(re).sum() for re in rewards]
        rewards = [np.array(re).mean() for re in rewards]

        success_count = []
        for success in successes:
            cur_success = 0
            for succ in success:
                cur_success += 1 if succ else 0
            success_count.append(cur_success)

        overall_success_steps = sum(success_count)

        over_all_steps = sum(steps)

        comp_rewards.append(rewards)
        comp_successes.append(success_count)

        print("Filename: ", filename)
        print("Steps: ", over_all_steps)
        print("Success steps: ", overall_success_steps)
        print("Overall rewards: ", np.array(overall_rewards).sum())
        print(
            "Average rewards: ", np.array(overall_rewards).sum() / overall_success_steps
        )

        # rewards = moving_average(rewards, windowsize)

    plot_datas(
        comp_rewards, f"{plot_dir}/{prefix}reward", smooth_window_size=None
    )
    plot_datas(
        comp_successes,
        f"{plot_dir}/{prefix}success_count",
        ylabel="Success Count",
    )
