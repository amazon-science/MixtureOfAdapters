# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import uuid
import pandas as pd
import os
from tqdm import tqdm


from mixture_of_adapters.data.bedrock_client import BedrockClient
from mixture_of_adapters.data.utils import read_jsonl_from_s3
import boto3


def parse_tasks(response: str):
    """
    Parses the output text of the prompt to extract document retrieval tasks.
    
    Args:
        text (str): The raw output containing <response>, <criteria>, and <task> tags.
        
    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary contains 'criteria' and 'task'.
    """
    tasks = []
    
    # Find all response blocks
    response_block = re.search(r"<response>(.*?)</response>", response, re.DOTALL)
    if not response_block:
        return tasks
    
    response_text = response_block.group(1).strip()
    criteria_match = re.search(r"<criteria>(.*?)</criteria>", response_text, re.DOTALL)
    task_matches = re.findall(r"<task>(.*?)</task>", response_text, re.DOTALL)
    
    if criteria_match and task_matches:
        for task in task_matches:
            tasks.append({
                "criteria": criteria_match.group(1).strip(),
                "task": task.strip()
            })

    return tasks
    

# Read the inference job output
def find_inference_job_id(s3_path, file_name="input_prompts.jsonl.out"):
    s3 = boto3.client("s3")
    bucket_name, prefix = s3_path.replace("s3://", "").split("/", 1)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    for obj in response.get("Contents", []):
        if obj["Key"].endswith(file_name):
            return f"s3://{bucket_name}/{obj['Key']}"  # Return the full URI of the file

    raise FileNotFoundError(f"File '{file_name}' not found in {s3_path}")


def parse_responses(responses):
    """
    Parses the responses from the model to extract triplet information.

    Args:
        responses (list): A list of response dictionaries from the model.
    
    Returns:
        list: A list of dictionaries containing triplet information.
    """
    all_obj = []
    for response in responses:
        modelOutput = response['modelOutput']
        if 'generation' in modelOutput:
            text = modelOutput['generation']
        elif 'content' in modelOutput:
            text = modelOutput['content'][0]['text']
        parsed_obj = parse_tasks(text)
        all_obj.extend(parsed_obj)
    return all_obj


if __name__ == "__main__":
    bucket_name = ""
    s3_inference_path = f"s3://{bucket_name}/embedding-adapter/data/products/tasks-1745390257/"

    # Read the inference job output
    job_output_path = find_inference_job_id(s3_inference_path)
    inference_outputs = read_jsonl_from_s3(s3_uri=job_output_path)

    tasks = parse_responses(inference_outputs)

    print(f"Parsed {len(tasks)} tasks from the inference job output.")

    # add a unique UUID to each task
    for task in tasks:
        task["id_task"] = str(uuid.uuid4())

    # save the tasks as pandas parquet file 
    tasks_s3_uri = os.path.join(s3_inference_path, "tasks.parquet")
    print(f"Saving tasks to {tasks_s3_uri}")
    pd.DataFrame(tasks).to_parquet(tasks_s3_uri, index=False)
