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
    get_query_from_memory,
    unpack_query_results,
    unpack_peek_results,
    update_collection,
)
from thesisexp.prompt import BasePrompt
from thesisexp.langchain_llava import LLaVA
from thesisexp.langchain_cogvlm import CogVLM

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment

from PIL import Image

from typing import List, Optional, Dict, Tuple

from langchain.llms.base import LLM
from collections import deque
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api import ClientAPI
import logging
import orjson

import random

import time
from copy import deepcopy


DEFAULT_IMAGE_TOKEN = "<image>"

# -----------------------------------------------------------------------------
# vector base utils
# -----------------------------------------------------------------------------


def list_collection_entries(collection: Collection):
    peek_result = collection.peek(limit=1000)

    count = 0
    success_count = 0
    for metadata, _ in zip(peek_result["metadatas"], peek_result["ids"]):
        print(
            f"task: {metadata['task']} {metadata['lang_goal']} is {metadata['success']}"
        )
        count += 1
        success_count += 1 if metadata["success"] else 0

    print("success rate: ", success_count / count)
    if len(peek_result["ids"]) != 1000:
        print("Total number of entries:", len(peek_result["ids"]))


def copy_collection(
    collection_from: str,
    collection_to: str,
    vec_store: ClientAPI,
    filter: Optional[Dict] = None,
    limit: int = 1000,
):
    collection_from: Collection = vec_store.get_collection(collection_from)

    collection_to: Collection = vec_store.get_or_create_collection(collection_to)

    if filter is None:
        peek_result = collection_from.peek(limit=limit)
        metadatas, _ = unpack_peek_results(peek_result)
    else:
        peek_result = collection_from.peek(limit=1)
        _, query_embedding = unpack_peek_results(peek_result)

        query_result = collection_from.query(
            query_embeddings=query_embedding, where=filter, n_results=limit
        )

        metadatas, _, _, _ = unpack_query_results(query_result)

    for metadata in metadatas:
        mem = MemEntry.from_dict(metadata)
        update_collection(mem, collection_to)


# -----------------------------------------------------------------------------
# LLM communication utils
# -----------------------------------------------------------------------------


def extract_instruction_from_response(response: str) -> str:
    # remove all before ":" smybol, including ":" and all after "." if exists
    # response = response.split(":")[1].split(".")[0]

    if ":" in response:
        response = response.split(":")[1]

    if "." in response:
        response = response.split(".")[0]

    return response


def get_memories(
    n_mems: int,
    embedding: List[float],
    collection: Collection,
    filters: List[Tuple[int, Optional[Dict]]],
    sample: bool = False,
) -> List[Optional[MemEntry]]:
    mems: List[Optional[MemEntry]] = []

    for num, filter in filters:
        get_num = num
        if sample:
            get_num *= 2
        query_result = collection.query(
            query_embeddings=[embedding],
            where=filter,
            n_results=get_num,
        )

        if query_result["metadatas"] is None:
            continue

        metadatas = query_result["metadatas"][0]

        for metadatas in metadatas:
            if metadatas is None:
                continue
            mem = MemEntry.from_dict(metadatas)
            mems.append(mem)

        if len(mems) > num:
            mems = random.sample(mems, num)

    if len(mems) < n_mems:
        query_result = collection.query(
            query_embeddings=[embedding],
            n_results=n_mems - len(mems),
        )

        metadatas, _, _, _ = unpack_query_results(query_result)

        for metadata in metadatas:
            mem = None
            if metadata is not None:
                mem = MemEntry.from_dict(metadata)

            mems.append(mem)

    elif len(mems) > n_mems:
        mems = mems[:n_mems]

    return mems


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


def feedback_agent_builder(agent_name: str, llm_server_url: str) -> LLM:
    if agent_name == "llava":
        agent = LLaVA()
        agent.set_url(llm_server_url)
        return agent
    elif agent_name == "cogvlm":
        agent = CogVLM()
        agent.set_url(llm_server_url)
        return agent
    else:
        raise ValueError(f"Unexpected feedback agent name: {agent_name}")


def correction_pipeline(
    correction_agent: LLM,
    obs: Dict[str, Tuple],
    lang_goal: str,
    instruction: str,
    chroma_collection: Collection,
    vcfg: Dict,
    step_log: Optional[Dict] = None,
) -> str:
    obs_image = Image.fromarray(np.array(obs["color"][1]))
    obs_image.resize((336, 336))
    obs_image.format = "PNG"

    curr_mem = MemEntry(
        lang_goal, instruction=instruction, images=[obs_image], task=vcfg["eval_task"]
    )

    embedding, _, _ = get_query_from_memory(curr_mem, use_begin=True)

    filters: List[Tuple[int, Optional[Dict]]] = []

    judge_filters: List[Tuple[int, Optional[Dict]]] = []

    judge_filters.append(
        (
            vcfg["correction_judge_n_examples"],
            {"task": vcfg["eval_task"]},
        )
    )

    judge_mems = get_memories(
        n_mems=vcfg["correction_judge_n_examples"],
        embedding=embedding,
        collection=chroma_collection,
        filters=judge_filters,
    )

    count_success = 0
    for mem in judge_mems:
        if mem is None:
            continue
        count_success += 1 if mem.success else 0

    success_rate = count_success / vcfg["correction_judge_n_examples"]

    goal_str = "Given the memory, this instruction is very likely to fail. Therefore we instead use this new command: "
    filters = [
        (
            vcfg["correction_n_examples"],
            {"task": {"$ne": vcfg["eval_task"]}},
        )
    ]

    if step_log is not None:
        step_log["queried_success_rate"] = success_rate
        step_log["given_instruction"] = instruction

    if not vcfg["exp_no_threshold"]:
        if success_rate > 0.6:
            if step_log is not None:
                step_log["enter high threshold"] = True

            goal_str = "Given the memory, this instruction is likely to success"
        else:
            if step_log is not None:
                step_log["enter low threshold"] = True
            goal_str = "Given the memory, this instruction is very likely to fail. Therefore we instead use this new command:"
            filters = [
                # {"success": True},
                # {"task": "block-insertion"},
                # {"$and": [{"task": vcfg["eval_task"]}, {"success": False}]},
                (
                    vcfg["correction_n_examples"] // 3,
                    {"$and": [{"task": vcfg["eval_task"]}, {"success": False}]},
                ),
                (
                    vcfg["correction_n_examples"] // 3,
                    {"$and": [{"task": vcfg["eval_task"]}, {"success": True}]},
                ),
                (
                    vcfg["correction_n_examples"] // 3,
                    {"task": {"$eq": vcfg["eval_task"]}},
                ),
            ]
        # else:
        #     if step_log is not None:
        #         step_log["enter mid threshold"] = True
        #     goal_str = "Given the memory, this instruction is possible to fail. Adding color information or locational information from our observation can be helpful. Therefore we use the improved instruction: "
            # filters = [
            #     # {"success": True},
            #     # {"task": "block-insertion"},
            #     # {"$and": [{"task": vcfg["eval_task"]}, {"success": False}]},
            #     (
            #         vcfg["correction_n_examples"] // 3,
            #         {"$and": [{"task": vcfg["eval_task"]}, {"success": False}]},
            #     ),
            #     (
            #         vcfg["correction_n_examples"] // 3,
            #         {"$and": [{"task": vcfg["eval_task"]}, {"success": True}]},
            #     ),
            #     (
            #         vcfg["correction_n_examples"] // 3,
            #         {"task": {"$eq": vcfg["eval_task"]}},
            #     ),
            # ]

    decided_instruction = instruction

    if not success_rate > 0.9 or vcfg["exp_no_threshold"]:
        mems = get_memories(
            n_mems=vcfg["correction_n_examples"],
            embedding=embedding,
            collection=chroma_collection,
            filters=filters,
        )

        # assert len(mems) == vcfg["correction_n_examples"], "Unexpected lens of memories"

        prompt = BasePrompt(
            task=vcfg["eval_task"],
            memories=mems,
            curr_mem=curr_mem,
            goal=goal_str,
            system_prompt="We are a robot agent doing table-top manipulation. The instruction will be fed to a model with limited capability for execution and we are trying to distinguish which language goal can successfully achieve its goal. similar instructions has similar success rate. ",
        )

        if step_log is not None:
            step_log["prompt"] = prompt.get_instruction_prompt(
                no_image_in_example=True, compact_curr=True
            )["prompt"]

        response: str = correction_agent(
            **prompt.get_instruction_prompt(no_image_in_example=True, compact_curr=True)
        )

        decided_instruction = extract_instruction_from_response(response)

    return decided_instruction


def correction_feedback_pipeline(
    correction_agent: LLM,
    obs_queue: deque,
    lang_goal: str,
    instruction: str,
    chroma_collection: Collection,
    vcfg: Dict,
):
    obs_images = [Image.fromarray(np.array(obs)) for obs in obs_queue]

    for img in obs_images:
        img.resize((336, 336))
        img.format = "PNG"

    curr_mem = MemEntry(
        lang_goal,
        instruction=instruction,
        images=obs_images,
        task=vcfg["eval_task"],
    )

    embedding, _, _ = get_query_from_memory(curr_mem, use_begin=True)

    filters: List[Tuple[int, Optional[Dict]]] = [
        # {"success": True},
        # {"task": "block-insertion"},
        # {"$and": [{"task": vcfg["eval_task"]}, {"success": False}]},
        (
            1 * vcfg["correction_feedback_n_examples"] // 3,
            {"$and": [{"task": vcfg["eval_task"]}, {"success": False}]},
        ),
        (
            1 * vcfg["correction_feedback_n_examples"] // 3,
            {"$and": [{"task": vcfg["eval_task"]}, {"success": True}]},
        ),
        (
            vcfg["correction_feedback_n_examples"] // 3,
            {"task": {"$eq": vcfg["eval_task"]}},
        ),
    ]

    mems = get_memories(
        n_mems=vcfg["correction_n_examples"],
        embedding=embedding,
        collection=chroma_collection,
        filters=filters,
    )

    prompt = BasePrompt(
        task=vcfg["eval_task"],
        memories=mems,
        curr_mem=curr_mem,
        goal="Given the current observation and memories, determine if current execution achieves the goal. We analyze the change of the target object's location from the image, and if similar instruction had good performance in examples. We ignore the change of the robot arm. Our reasoning is: ",
        system_prompt="We are a robot agent observing table-top manipulation. The language goal will be fed to a model with limited capability for execution and we are trying to distinguish which language goal can successfully achieve its described goal. Similar language goals has similar success rate. ",
    )

    response: str = correction_agent(
        **prompt.get_instruction_prompt(no_image_in_example=True, compact_curr=False)
    )

    prompt.add_prompt(response)

    prompt.add_prompt(
        "Gien your reasoning, reply 'true' if success or 'false' if fails. Response: "
    )

    yes_response: str = correction_agent(
        **prompt.get_instruction_prompt(no_image_in_example=True, compact_curr=False)
    )

    while "true" not in yes_response.lower() and "false" not in yes_response.lower():
        yes_response: str = correction_agent(
            **prompt.get_instruction_prompt(
                no_image_in_example=True, compact_curr=False
            )
        )

    success = False
    if "true" in yes_response.lower():
        success = True
    elif "false" in yes_response.lower():
        success = False

    curr_mem.success = success

    return curr_mem


def feedback_pipeline(
    feedback_agent: LLM,
    obs_queue: deque,
    lang_goal: str,
    instruction: str,
    chroma_collection: Collection,
    vcfg: Dict,
) -> Tuple[MemEntry, str]:
    obs_images = [Image.fromarray(np.array(obs)) for obs in obs_queue]

    for img in obs_images:
        img.resize((336, 336))
        img.format = "PNG"

    curr_mem = MemEntry(
        lang_goal=lang_goal,
        instruction=instruction,
        images=obs_images,
        task=vcfg["eval_task"],
    )

    embedding, _, prompt = get_query_from_memory(
        curr_mem, url=vcfg["llm_embedding_url"]
    )

    filters: List[Tuple[int, Dict]] = [
        (2, {"task": {"$eq": vcfg["eval_task"]}}),
        (1, {"task": {"$ne": vcfg["eval_task"]}}),
    ]

    filters = []

    mems = get_memories(
        n_mems=vcfg["feedback_n_examples"],
        embedding=embedding,
        collection=chroma_collection,
        filters=filters,
    )

    prompt = BasePrompt(
        task=vcfg["eval_task"],
        memories=mems,
        curr_mem=curr_mem,
    )

    assert feedback_agent is not None, "feedback_agent is None"

    response: str = feedback_agent(
        **prompt.get_instruction_prompt(no_image_in_example=True, compact_curr=True)
    )

    feedback = response[:300] if len(response) < 200 else response

    prompt.add_prompt(response)
    prompt.add_prompt(
        "Given the answer, respose with 'True' for success or 'False' for unsuccess. Response:"
    )

    yes_response: str = feedback_agent(
        **prompt.get_memory_prompt(compact_curr=vcfg["compact_curr_mem"])
    )

    while "true" not in yes_response.lower() and "false" not in yes_response.lower():
        yes_response: str = feedback_agent(
            **prompt.get_instruction_prompt(no_image_in_example=True, compact_curr=True)
        )

    success = False
    if "true" in response.lower():
        success = True
    elif "false" in response.lower():
        success = False

    curr_mem.success = success

    return curr_mem, feedback


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

        log_dict = None
        log_file = None

        if vcfg["compare_logging"]:
            log_dict: Dict = {}

            name_suffixs: List[str] = [
                vcfg["eval_task"],
                "feedback" if vcfg["feedback"] else "",
                vcfg["feedback_agent"] if vcfg["feedback"] else "",
                vcfg["correction_feedback_agent"]
                if vcfg["correction_feedback"]
                else "",
                f"correction_{vcfg['correction_n_examples']}"
                if vcfg["correction"]
                else "",
                "correction_feedback" if vcfg["correction_feedback"] else "",
                f"{vcfg['n_demos']}_demos",
                "no_threshold" if vcfg["exp_no_threshold"] else "",
                time.strftime("%Y-%m-%d-%H", time.localtime()),
            ]

            name_suffixs = [s for s in name_suffixs if s != ""]

            log_filename: str = "-".join(name_suffixs) + ".json"

            log_file = os.path.join(vcfg["compare_logging_path"], log_filename)

            log_dict["task"] = vcfg["eval_task"]
            log_dict["start_time"] = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime()
            )

            log_dict["feedback"] = vcfg["feedback"]
            log_dict["correction"] = vcfg["correction"]
            log_dict["correction_feedback"] = vcfg["correction_feedback"]
            log_dict["correction_n_examples"] = vcfg["correction_n_examples"]
            log_dict["feedback_n_examples"] = vcfg["feedback_n_examples"]
            log_dict["compact_curr_mem"] = vcfg["compact_curr_mem"]
            log_dict["vector_base"] = vcfg["vector_base_source"]

        feedback_agent = None
        if vcfg["feedback"]:
            feedback_agent = feedback_agent_builder(
                vcfg["feedback_agent"], vcfg["llm_server_url"]
            )

        correction_feedback_agent = None
        if vcfg["correction_feedback"]:
            correction_feedback_agent = feedback_agent_builder(
                vcfg["correction_feedback_agent"], vcfg["llm_server_url"]
            )

        correction_agent = None
        if vcfg["correction"]:
            correction_agent = feedback_agent_builder(
                vcfg["correction_agent"], vcfg["llm_server_url"]
            )

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

            collection_name = vcfg["vector_base"]
            vec_store = chromadb.PersistentClient(path="./chroma_db")

            # ls = vec_store.list_collections()

            # known issue here: assume the vector_base collection exists
            if vcfg["vector_base_control"] and vcfg["correction_feedback"]:
                vec_store.delete_collection(collection_name)

                copy_collection(vcfg["vector_base_source"], collection_name, vec_store)

            chroma_collection = vec_store.get_or_create_collection(collection_name)

            # avoid potential out of bound
            back_lang_goal = []

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

                # implementation for seperate language goal from intention.
                # Assumption: the linear movement of language goals

                obs = env.reset()

                original_language_goal = []
                # if not len(env.task.lang_goals) == 0:
                #     original_language_goal = env.task.lang_goals

                if i == 0:
                    original_language_goal = deepcopy(env.task.lang_goals)
                    back_lang_goal = deepcopy(env.task.lang_goals)
                else:
                    original_language_goal = deepcopy(back_lang_goal)

                info = env.info
                reward = 0

                epoch_log_dict = None
                if vcfg["compare_logging"]:
                    log_dict[f"epoch_{i}"] = {}
                    epoch_log_dict = log_dict[f"epoch_{i}"]

                # Start recording video (NOTE: super slow)
                if record:
                    video_name = f"{task_name}-{i+1:06d}"
                    if "multi" in vcfg["model_task"]:
                        video_name = f"{vcfg['model_task']}-{video_name}"
                    env.start_rec(video_name)

                obs_queue = deque(maxlen=2)
                obs_queue.append(obs["color"][0])

                feedback = ""

                if vcfg["compare_logging"]:
                    assert epoch_log_dict is not None, "epoch_log_dict is None"
                    epoch_log_dict["num_mem"] = chroma_collection.count()
                    epoch_log_dict["vb_name"] = collection_name

                for i, _ in enumerate(range(task.max_steps)):
                    lang_goal = info["lang_goal"]

                    print(f"Lang Goal: {lang_goal}")

                    step_log_dict = None
                    if vcfg["compare_logging"]:
                        assert epoch_log_dict is not None, "epoch_log_dict is None"

                        epoch_log_dict[f"step_{i}"] = {}
                        step_log_dict = epoch_log_dict[f"step_{i}"]

                    if vcfg["correction"]:
                        decided_instruction = correction_pipeline(
                            correction_agent=correction_agent,
                            obs=obs,
                            instruction=info["lang_goal"],
                            lang_goal=original_language_goal[0],
                            chroma_collection=chroma_collection,
                            vcfg=vcfg,
                            step_log=step_log_dict,
                        )

                        # chop the instruction if it exceeds the limit of 77 tokens
                        if len(decided_instruction.split(" ")) > 70:
                            decided_instruction = " ".join(
                                decided_instruction.split(" ")[:70]
                            )

                        env.task.lang_goals[0] = decided_instruction

                        info["new instruction"] = decided_instruction

                    act = agent.act(obs, info, goal)

                    obs, reward, done, info = env.step(action=act, feedback=feedback)

                    obs_queue.append(obs["color"][0])

                    if vcfg["compare_logging"]:
                        assert step_log_dict is not None, "step_log_dict is None"
                        if isinstance(reward, np.float64):
                            step_log_dict["reward"] = reward.item()
                        else:
                            step_log_dict["reward"] = reward

                        if isinstance(done, np.bool_):
                            step_log_dict["done"] = done.item()
                        else:
                            step_log_dict["done"] = done

                        step_log_dict["lang_goal"] = info["lang_goal"]

                        assert log_dict is not None, "log_dict is None, unexpected"
                        assert log_file is not None, "log_file is None, unexpected"

                        with open(log_file, "wb") as f:
                            f.write(orjson.dumps(log_dict) + b"\n")

                    # display current observation in cli
                    if vcfg["cli_img"]:
                        utils.display_image_in_cli(
                            [Image.fromarray(np.array(obs)) for obs in list(obs_queue)]
                        )

                    if len(original_language_goal) < 1:
                        # catch point to debug the index error occurs randomly
                        breakpoint()

                    if vcfg["correction_feedback"]:
                        curr_mem = correction_feedback_pipeline(
                            correction_agent=correction_feedback_agent,
                            obs_queue=obs_queue,
                            lang_goal=original_language_goal[0],
                            instruction=info["lang_goal"],
                            chroma_collection=chroma_collection,
                            vcfg=vcfg,
                        )

                        if vcfg["compare_logging"]:
                            gt_success = True if reward > 0 else False

                            assert step_log_dict is not None, "step_log_dict is None"

                            step_log_dict["correct_prediction"] = (
                                True if gt_success == curr_mem.success else False
                            )

                            logging.info(
                                f"Evaluation on step {i}: success {curr_mem.success}. Correct evaluation: {gt_success == curr_mem.success}"
                            )

                            if vcfg["correction_feedback_use_gt_label"]:
                                curr_mem.success = gt_success

                        add_memory_into_collection(
                            chroma_collection,
                            curr_mem,
                            embedding_url=vcfg["llm_embedding_url"],
                        )

                    if vcfg["feedback"]:
                        # trasnform current_information in the memory

                        curr_mem, feedback = feedback_pipeline(
                            feedback_agent=feedback_agent,
                            obs_queue=obs_queue,
                            lang_goal=original_language_goal[0],
                            instruction=info["lang_goal"],
                            chroma_collection=chroma_collection,
                            vcfg=vcfg,
                        )

                        logging.info(
                            f"Evaluation on step {i}: success {curr_mem.success}, reason: {feedback}"
                        )

                        gt_success = True if reward > 0 else False

                        if vcfg["feedback_use_gt_label"] and vcfg["compare_logging"]:
                            curr_mem.success = True if reward > 0 else False

                            assert step_log_dict is not None, "step_log_dict is None"

                            step_log_dict["feedback_prediction"] = (
                                gt_success == curr_mem.success
                            )
                            step_log_dict["feedback"] = feedback

                        add_memory_into_collection(chroma_collection, curr_mem)

                    # obs_queue.popleft();
                    total_reward += reward

                    # implement for seperate language goal from intention

                    print(f"Total Reward: {total_reward:.3f} | Done: {done}\n")
                    if done:
                        env.add_video_end_frame()
                        env.last_frame = None
                        break
                # handle the last frame if max_steps is reached

                if env.last_frame is not None:
                    env.add_video_end_frame()

                if vcfg["compare_logging"]:
                    assert epoch_log_dict is not None, "epoch_log_dict is None"

                    if isinstance(total_reward, np.float64):
                        epoch_log_dict["total_reward"] = total_reward.item()
                    else:
                        epoch_log_dict["total_reward"] = total_reward

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
        if vcfg["compare_logging"]:
            assert log_dict is not None, "log_dict is None, unexpected"
            assert log_file is not None, "log_file is None, unexpected"

            with open(log_file, "wb") as f:
                f.write(orjson.dumps(log_dict) + b"\n")

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
