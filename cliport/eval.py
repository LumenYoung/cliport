"""Ravens main training script."""

import io
import os
import json
from thesisexp.memory.MemoryStorage import MemEntry, NaiveMemStorage
from thesisexp.memory.Memory import NaiveMemory, BaseMemory
from thesisexp.utils import (
    add_memory_into_collection,
    transform_mem_to_prompt,
    add_image_to_file,
    get_embedding_from_llava,
    get_query_from_memory,
    unpack_result,
)
from thesisexp.prompt import BasePrompt
from thesisexp.langchain_llava import LLaVA

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment

from PIL import Image

from typing import Any, List, Optional, Dict, Tuple

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from collections import deque
import chromadb
import logging


DEFAULT_IMAGE_TOKEN = "<image>"

# -----------------------------------------------------------------------------
# LLM communication utils
# -----------------------------------------------------------------------------


def save_array_to_image(array, filename):
    """
    Convert a numpy array to an image and save it to a file.

    Parameters:
    array (numpy.ndarray): The numpy array to convert.
    filename (str): The name of the file to save the image to.
    """
    # Convert the array to an image
    img = Image.fromarray(array.astype("uint8"))

    # Save the image
    img.save(filename)


def image_to_byte_array(image: Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def llava_few_shot_feedback(
    curr_mem: MemEntry, llm: LLaVA, memory: BaseMemory
) -> Tuple[str, bool]:
    file = {}

    good_example, bad_example = memory.sample("insert-block")

    mem_good_prompt, images_1 = transform_mem_to_prompt(good_example)

    add_image_to_file(file, images_1)

    mem_bad_prompt, images_2 = transform_mem_to_prompt(bad_example)

    add_image_to_file(file, images_2)

    curr_prompt, images_3 = transform_mem_to_prompt(curr_mem)

    add_image_to_file(file, images_3)

    system_prompt = "You are trying to describe a robot manipulation outcome."

    prompt = f"""
    system: {system_prompt}

    example 1:
    {mem_good_prompt}

    example 2:
    {mem_bad_prompt}

    current observation:
    {curr_prompt}

    Reply with your reasoning whether the goal is achieved or not.
    Keep all descriptions short and relavant.

    answer:
    """

    # discarded Describe whether the goal is achieved or not.

    response1 = llm(prompt, images=file)

    prompt += response1

    prompt += "\nGiven the answer, respose with 'True' for success or 'False' for unsuccess. Response:"

    response2 = llm(prompt)

    if "true" in response2.lower():
        return response2, True
    elif "false" in response2.lower():
        return response2, False
    else:
        raise Exception(f"Unexpected response: {response2}")


def llava_feedback(images: tuple, llm: LLaVA):
    prompt: str = f"""
    system: In the image {DEFAULT_IMAGE_TOKEN} is a robot several objects on the table,
    in {DEFAULT_IMAGE_TOKEN} is the outcome.

    goal:
        Describe the color and shape of the object starting with "object:".
        Describe the motion (most importantly direction in left, right, up, down) of the robot starting with "motion:".
        Keep all descriptions short and simple. Reply with nothing irrelevant.

    answer:
    """
    result: Dict = {}

    img1 = Image.fromarray(np.array(images[0])).convert("RGB")
    img1.format = "PNG"
    # img1.info = {'dpi': (96.012, 96.012)}
    img1.is_animated = False
    img1.n_frames = 1
    img1.resize((336, 336))

    img2 = Image.fromarray(np.array(images[1])).convert("RGB")
    img2.format = "PNG"
    # img2.info = {'dpi': (96.012, 96.012)}
    img2.is_animated = False
    img2.n_frames = 1
    img1.resize((336, 336))

    # img1.save("dummy.png")

    # with open("dummy.png", "rb") as img_file:
    #     result["image_file_1"] = img_file.read()

    result["image_file_1"] = image_to_byte_array(img1)
    # img2.save("dummy.png")

    # with open("dummy.png", "rb") as img_file:
    #     result["image_file_2"] = img_file.read()

    result["image_file_2"] = image_to_byte_array(img2)

    return llm(prompt, images=result)


# ################################# DEBUG CODE

# images = obs["color"]
# output = llava_feedback(images)

# print(type(images))

# p np.array(obs["color"][0]).shape
# p Image.fromarray(np.array(obs["color"][0]))

# img = Image.fromarray(np.array(obs["color"][0]))
# img.format="PNG"
# img_bytes = image_to_byte_array(img)
# result = {}
# result["image_file_1"] = img_bytes
# files = {name: (name, img, 'image/png') for name, img in result.items()}
# url = "http://experiment_env:6000/conversation"
# prompt = "describe the image <image>"
# response = requests.post(url, files=files, data={"prompt": prompt})


# print(Image.fromarray(np.array(images[0])).convert("RGB").mode)
###################################


@hydra.main(config_path="./cfg", config_name="eval")
def main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg["train_config"])

    llm = LLaVA()
    # Initialize environment and task.
    env = Environment(
        vcfg["assets_root"],
        disp=vcfg["disp"],
        shared_memory=vcfg["shared_memory"],
        hz=480,
        record_cfg=vcfg["record"],
    )

    # Choose eval mode and task.
    mode = vcfg["mode"]
    eval_task = vcfg["eval_task"]
    if mode not in {"train", "val", "test"}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg["type"]
    if "multi" in dataset_type:
        ds = dataset.RavensMultiTaskDataset(
            vcfg["data_dir"],
            tcfg,
            group=eval_task,
            mode=mode,
            n_demos=vcfg["n_demos"],
            augment=False,
        )
    else:
        ds = dataset.RavensDataset(
            os.path.join(vcfg["data_dir"], f"{eval_task}-{mode}"),
            tcfg,
            n_demos=vcfg["n_demos"],
            augment=False,
        )

    all_results = {}
    name = "{}-{}-n{}".format(eval_task, vcfg["agent"], vcfg["n_demos"])

    # Save path for results.
    json_name = (
        f"multi-results-{mode}.json"
        if "multi" in vcfg["model_path"]
        else f"results-{mode}.json"
    )
    save_path = vcfg["save_path"]
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f"{name}-{json_name}")

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, "r") as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg["model_path"], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg["update_results"] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        results = []
        mean_reward = 0.0

        # Run testing for each training run.
        for train_run in range(vcfg["n_repeats"]):
            # Initialize agent.
            utils.set_seed(train_run, torch=True)
            agent = agents.names[vcfg["agent"]](name, tcfg, None, ds)

            # Load checkpoint
            agent.load(model_file)
            print(f"Loaded: {model_file}")

            record = vcfg["record"]["save_video"]
            n_demos = vcfg["n_demos"]

            # Run testing and save total rewards with last transition info.
            for i in range(0, n_demos):
                print(f"Test: {i + 1}/{n_demos}")
                episode, seed = ds.load(i)
                goal = episode[-1]
                total_reward = 0
                np.random.seed(seed)

                # set task
                if "multi" in dataset_type:
                    task_name = ds.get_curr_task()
                    task = tasks.names[task_name]()
                    print(f"Evaluating on {task_name}")
                else:
                    task_name = vcfg["eval_task"]
                    task = tasks.names[task_name]()

                task.mode = mode
                env.seed(seed)
                env.set_task(task)
                obs = env.reset()
                info = env.info
                reward = 0

                # Start recording video (NOTE: super slow)
                if record:
                    video_name = f"{task_name}-{i+1:06d}"
                    if "multi" in vcfg["model_task"]:
                        video_name = f"{vcfg['model_task']}-{video_name}"
                    env.start_rec(video_name)

                obs_queue = deque(maxlen=2)
                obs_queue.append(obs["color"][0])

                feedback = ""
                # initialize the memory

                # memory = NaiveMemory(NaiveMemStorage(vcfg["memory_storage_dir"]))

                vec_store = chromadb.PersistentClient(path="./chroma_db")

                collection_name = "primary"

                chroma_collection = vec_store.get_or_create_collection(collection_name)

                for i, _ in enumerate(range(task.max_steps)):
                    act = agent.act(obs, info, goal)
                    lang_goal = info["lang_goal"]
                    print(f"Lang Goal: {lang_goal}")
                    obs, reward, done, info = env.step(action=act, feedback=feedback)
                    obs_queue.append(obs["color"][0])

                    # display current observation in cli
                    if vcfg["cli_img"]:
                        utils.display_image_in_cli(
                            [Image.fromarray(np.array(obs)) for obs in list(obs_queue)]
                        )

                    if vcfg["feedback"]:
                        # trasnform current_information in the memory

                        obs_images = [
                            Image.fromarray(np.array(obs)) for obs in obs_queue
                        ]

                        for img in obs_images:
                            img.resize((336, 336))
                            img.format = "PNG"

                        curr_mem = MemEntry(
                            lang_goal,
                            images=obs_images,
                            task=vcfg["eval_task"],
                        )

                        embedding, metadata, prompt = get_query_from_memory(curr_mem)

                        success_filter = {"success": {"$eq": True}}

                        success_query = chroma_collection.query(
                            query_embeddings=[embedding],
                            where=success_filter,
                            n_results=1,
                        )

                        succ_metadata, _, _ = unpack_result(success_query)

                        mem_succ = None
                        if succ_metadata is not None:
                            mem_succ = MemEntry.from_dict(succ_metadata)

                        failure_filter = {"success": {"$eq": False}}
                        failure_query = chroma_collection.query(
                            query_embeddings=[embedding],
                            where=failure_filter,
                            n_results=1,
                        )
                        fail_metadata, _, _ = unpack_result(failure_query)

                        mem_fail = None
                        if fail_metadata is not None:
                            mem_fail = MemEntry.from_dict(fail_metadata)

                        prompt = BasePrompt(
                            task=vcfg["eval_task"],
                            memories=[mem_fail, mem_succ],
                            curr_mem=curr_mem,
                        )

                        response: str = llm(
                            **prompt.get_instruction_prompt(compact_example=True)
                        )

                        feedback = response[:300] if len(response) < 200 else response

                        prompt.add_prompt(response)
                        prompt.add_prompt(
                            "Given the answer, respose with 'True' for success or 'False' for unsuccess. Response:"
                        )

                        response: str = llm(**prompt.get_memory_prompt())

                        success = False
                        if "true" in response.lower():
                            success = True
                        elif "false" in response.lower():
                            success = False
                        else:
                            raise Exception(f"Unexpected response: {response}")

                        curr_mem.success = success

                        logging.info(
                            f"Evaluation on step {i}: success {success}, reason: {feedback}"
                        )

                        add_memory_into_collection(chroma_collection, curr_mem)

                    # obs_queue.popleft();
                    total_reward += reward
                    print(f"Total Reward: {total_reward:.3f} | Done: {done}\n")
                    if done:
                        env.add_video_end_frame()
                        env.last_frame = None
                        break

                # handle the last frame if max_steps is reached
                if env.last_frame is not None:
                    env.add_video_end_frame()

                results.append((total_reward, info))
                mean_reward = np.mean([r for r, _ in results])
                print(f"Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}")

                # End recording video
                if record:
                    env.end_rec()

            all_results[ckpt] = {
                "episodes": results,
                "mean_reward": mean_reward,
            }

        # Save results in a json file.
        if vcfg["save_results"]:
            # Load existing results
            if os.path.exists(save_json):
                with open(save_json, "r") as f:
                    existing_results = json.load(f)
                existing_results.update(all_results)
                all_results = existing_results

            with open(save_json, "w") as f:
                json.dump(all_results, f, indent=4)


def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg["checkpoint_type"] == "last":
        last_ckpt = "last.ckpt"
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg["checkpoint_type"] == "val_missing":
        checkpoints = sorted(
            [c for c in os.listdir(vcfg["model_path"]) if "steps=" in c]
        )
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg["checkpoint_type"] == "test_best":
        result_jsons = [
            c for c in os.listdir(vcfg["results_path"]) if "results-val" in c
        ]
        if "multi" in vcfg["model_task"]:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg["results_path"], result_json), "r") as f:
                eval_res = json.load(f)
            best_checkpoint = "last.ckpt"
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res["mean_reward"] > best_success:
                    best_checkpoint = ckpt
                    best_success = res["mean_reward"]
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = "last.ckpt"
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [
            c for c in os.listdir(vcfg["model_path"]) if vcfg["checkpoint_type"] in c
        ]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


if __name__ == "__main__":
    # llava_test()

    main()
