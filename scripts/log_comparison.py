import os
from pathlib import Path
import orjson
from typing import Dict, List, Tuple, Optional
import numpy as np
from chromadb.api.models.Collection import Collection

import matplotlib

matplotlib.use("Agg")
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
            curr_epoch_success.append(True if step_value["reward"] > 0 else False)

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
        plt.plot(data, label=file_name)  # Plot the rewards

    plt.xlabel(xlabel)  # Set the x-axis label
    plt.ylabel(ylabel)  # Set the y-axis label

    dirname = os.path.dirname(file_name)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    plt.savefig(file_name)


def extract_feedback_prediction(data: Dict) -> List[bool]:
    result = []

    for key, value in data.items():
        if not key.startswith("epoch_"):
            continue

        for step_key, step_value in value.items():
            if not step_key.startswith("step_"):
                continue
            if "feedback_prediction" in step_value.keys():
                result.append(step_value["feedback_prediction"])
            elif "correct_prediction" in step_value.keys():
                result.append(step_value["correct_prediction"])

    return result


def calculate_experiments_and_plot():
    log_dir = "./logs"
    plot_dir = "./plots"

    log_files = os.listdir(log_dir)

    log_files = {
        # # "llava palletizing-boxes old 2":"palletizing-boxes-correction_5examples-correction_feedback-100_demos-2023-12-18.json",
        # # # "towers-of-hanoi-seq-full-100_demos-2023-12-17.json",
        # # # "towers-of-hanoi-seq-full-correction_5examples-100_demos-2023-12-18.json",
        # # "llava block-insertion old":"block-insertion-correction_5examples-correction_feedback-100_demos-2023-12-19.json",
        # # "block insertion vanilla":"block-insertion-100_demos-2023-12-16.json",
        # "vanilla palletizing-boxes": "palletizing-boxes-100_demos-2023-12-16.json",
        # # "llava palletizing-boxes old": "palletizing-boxes-correction_5examples-100_demos-2023-12-16.json",
        # # "correction and feedback no thresholding": "palletizing-boxes-correction_5examples-correction_feedback-100_demos-no_threshold-2023-12-26.json",
        # "correction and feedback thresholded": "palletizing-boxes-correction_5examples-correction_feedback-100_demos-2023-12-18.json",
        # # # "palletizing-boxes-feedback-cogvlm-50_demos-2024-01-08.json",
        # # # "palletizing-boxes-feedback-llava-50_demos-2024-01-09.json",
        # # "cogvlm new exp":"block-insertion-cogvlm-correction_5examples-correction_feedback-50_demos-2024-01-13.json",
        # # "llava new exp":"block-insertion-llava-correction_feedback-50_demos-2024-01-13.json",
        # "llava low thresholding palletizing box": "palletizing-boxes-llava-correction_5examples-correction_feedback-20_demos-no_threshold-2024-01-15.json",
        # "llava two thresholding pallatizing box": "palletizing-boxes-llava-correction_5examples-correction_feedback-20_demos-2024-01-15.json",
        # # "llava three thresholding pallatizing box": "palletizing-boxes-llava-correction_5examples-correction_feedback-20_demos-2024-01-16.json",
        # "llava 0.5 0.9 three thresholding pallatizing box": "palletizing-boxes-llava-correction_5examples-correction_feedback-20_demos-2024-01-16.json",
        # "llava newer experiment with more examples": "palletizing-boxes-llava-correction_7-correction_feedback-20_demos-2024-01-16-16.json",
        # "llava block-insertion seperate intention from instruction": "block-insertion-llava-correction_5-correction_feedback-20_demos-2024-01-22-00.json",
        # "llava pallatizing-box seperate intention from instruction": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-22-09.json",
        # "llava correct pallatizing-box": "palletizing-boxes-llava-correction_3-correction_feedback-20_demos-2024-01-22-17.json",
        # "llava with true and false filter": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-22-23.json",
        # "llava with all false code base, same setup from previous": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-23-09.json",
        # "llava like previous, but with sampling disabled": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-23-11.json",
        # "llava from previous, but using the correct vector base": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-23-18.json",
        # "llava from previous, but using low threshold prompt": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-no_threshold-2024-01-23-22.json",
        # "llava low thresholding using the all fail vector base": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-no_threshold-2024-01-23-22.json",
        # "llava second small tryout in 3 epoch with ": "palletizing-boxes-llava-correction_5-correction_feedback-3_demos-2024-01-26-15.json",
        # "llava new try with correct base": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-26-23.json",
        # "llava same as previous, but longer": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-27-23.json",
        "llava with two thresholds": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-28-23.json",
        "llava with two threshold, but lower the bar": "palletizing-boxes-llava-correction_5-correction_feedback-5_demos-2024-01-29-11.json",
        "llava with two threshold, also expand the query size": "palletizing-boxes-llava-correction_5-correction_feedback-5_demos-2024-01-29-16.json",
        "previous exp with more more demos": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-30-09.json",
        "tryout with new prompt at Jan 30": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-30-15.json",
        "tryout new prompt and new threshold at Jan 31": "palletizing-boxes-llava-correction_5-correction_feedback-20_demos-2024-01-31-09.json",
        "simplified work at Jan 31": "palletizing-boxes-llava-correction_4-correction_feedback-20_demos-no_threshold-2024-01-31-17.json",
        "simpiified pipeline as well as image for the prompt": "palletizing-boxes-llava-correction_2-correction_feedback-20_demos-no_threshold-2024-01-31-19.json"
    }

    plt.figure(figsize=(10, 6))  # Set the figure size

    windowsize = 1

    comp_rewards = []
    comp_successes = []

    prefix = longest_common_prefix(log_files)

    for label, filename in log_files.items():
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

        print("Experiment: ", label)
        print("   Filename: ", filename)
        print("   Steps: ", over_all_steps)
        print("   Success steps: ", overall_success_steps)
        print("   Overall rewards: ", np.array(overall_rewards).sum())
        if overall_success_steps > 0:
            print(
                "   Average rewards: ",
                np.array(overall_rewards).sum() / over_all_steps,
            )

        # rewards = moving_average(rewards, windowsize)

    response = input(f"Filename {prefix}? [y/n]")

    if response == "n":
        prefix = input("Enter the prefix:")

    plot_datas(
        comp_rewards, f"{plot_dir}/{prefix}reward", smooth_window_size=windowsize
    )
    plot_datas(
        comp_successes,
        f"{plot_dir}/{prefix}success_count",
        ylabel="Success Count",
    )


def compare_feedback_accuracy():
    log_files: Dict = {
        "cogvlm_palletizing": "palletizing-boxes-feedback-cogvlm-20_demos-2024-01-10.json",
        "llava_palletizing": "palletizing-boxes-feedback-cogvlm-20_demos-2024-01-10.json",
        "cogvlm_block_insertion": "block-insertion-feedback-cogvlm-20_demos-2024-01-10.json",
        "llava_block_insertion": "block-insertion-feedback-llava-20_demos-2024-01-10.json",
        "distinct sample llava feedback palletizing-boxes": "palletizing-boxes-llava-correction_5examples-correction_feedback-20_demos-2024-01-12.json",
        "distinct sample cogvlm feedback palletizing-boxes": "palletizing-boxes-cogvlm-correction_5examples-correction_feedback-20_demos-2024-01-11.json",
        "distinct sample cogvlm feedback block-insertion": "block-insertion-cogvlm-correction_feedback-20_demos-2024-01-12.json",
        "distinct sample llava feedback block-insertion": "block-insertion-llava-correction_feedback-20_demos-2024-01-12.json",
    }

    #  block-insertion-feedback-cogvlm-20_demos-2024-01-10.json
    #  block-insertion-feedback-llava-20_demos-2024-01-10.json

    log_dir = "./logs"

    feedback_predictions_list = []

    data_list = []

    for key, value in log_files.items():
        data = load_json_log(Path(log_dir, value))

        data_list.append(data)

        feedback_predictions = extract_feedback_prediction(data)

        total = len(feedback_predictions)
        correct = sum(feedback_predictions)

        feedback_predictions_list.append(feedback_predictions)

        print(f"{key}: {correct}/{total} = {correct/total}")

    for pred_1, pred_2 in zip(
        feedback_predictions_list[0], feedback_predictions_list[1]
    ):
        if pred_1 != pred_2:
            print("Different")

    # breakpoint()

    # print("Hello")


if __name__ == "__main__":
    calculate_experiments_and_plot()
    # compare_feedback_accuracy()
