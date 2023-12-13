import os
from pathlib import Path
import orjson
from typing import Dict, List


def load_json_log(filepath: Path):
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


if __name__ == "__main__":
    log_dir = "./logs"

    log_files = os.listdir(log_dir)

    for filename in log_files:
        data = load_json_log(Path(log_dir, filename))

        rewards = extract_step_rewards(data)
        success = extract_step_success(data)

        print(rewards)
        print(success)
