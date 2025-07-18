# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import uuid
import pandas as pd
import os
from tqdm import tqdm
import boto3

from mixture_of_adapters.data.utils import read_jsonl_from_s3


def parse_triplet(response: str):
    """
    Parses the response string to extract triplet information.

    Args:
        response (str): The response string containing triplet information.

    Returns:
        Dict[str, str]: A dictionary containing the query document, negative document, and positive document.
    """
    try:
        # Find all response blocks
        response_blocks = re.findall(r"<example>.*?(\{.*?\}).*?</example>", response, re.DOTALL)
        if not response_blocks:
            return
        triplet = json.loads(response_blocks[0])
        if "anchorSentence" not in triplet or "negativeSentence" not in triplet or "positiveSentence" not in triplet:
            return
        if triplet["anchorSentence"] is None or triplet["negativeSentence"] is None or triplet["positiveSentence"] is None:
            return
        return {
            "anchor_document": triplet["anchorSentence"],
            "id_anchor_document": str(uuid.uuid4()),
            "negative_document": triplet["negativeSentence"],
            "id_negative_document": str(uuid.uuid4()),
            "positive_document": triplet["positiveSentence"],
            "id_positive_document": str(uuid.uuid4()),
        }
    except json.JSONDecodeError:
        return
    
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
    all_triplets = []
    for response in responses:
        modelOutput = response['modelOutput']
        if 'generation' in modelOutput:
            triplet_text = modelOutput['generation']
        elif 'content' in modelOutput:
            triplet_text = modelOutput['content'][0]['text']
        triplet = parse_triplet(triplet_text)
        record_id = response['recordId']
        if triplet:
            triplet['recordId'] = record_id
            all_triplets.append(triplet)
    return all_triplets


if __name__ == "__main__":
    bucket_name = ""
    s3_inference_paths = [
        f"s3://{bucket_name}/embedding-adapter/data/csts/triplets-1744760875/",
    ]

    # Read and concatenate prompt records
    triplets_df_list = []
    for inf_path in s3_inference_paths:
        print(f"Processing inference path: {inf_path}")

        prompt_records = pd.read_parquet(os.path.join(inf_path, "prompt_records.parquet"))
        print(f"Read {len(prompt_records)} prompt records.")
        print(f"Prompt records columns: {prompt_records.columns.tolist()}")

        job_output_path = find_inference_job_id(inf_path)
        inference_outputs = read_jsonl_from_s3(s3_uri=job_output_path)
        triplets = parse_responses(inference_outputs)
        triplets_df = pd.DataFrame(triplets)
        triplets_df = pd.merge(prompt_records, triplets_df, on="recordId", how="inner").drop(columns=["recordId"])
        print(f"Parsed {len(triplets_df)} triplets.")
        triplets_df_list.append(triplets_df)

    # Concatenate all triplet dataframes
    merged_df = pd.concat(triplets_df_list, axis=0, ignore_index=True)
    merged_df = merged_df.rename(columns={"condition": "task"})

    # Assign a unique UUID for each unique value in the 'task' column
    task_to_uuid = {task: str(uuid.uuid4()) for task in merged_df["task"].unique()}
    merged_df["id_task"] = merged_df["task"].map(task_to_uuid)
    
    print(f"Merged df columns: {merged_df.columns.tolist()}")

    if len(s3_inference_paths) == 1:
        triplets_uri = os.path.join(s3_inference_paths[0], "triplets.parquet")
    else:
        triplets_uri = os.path.join(s3_inference_paths[0].rstrip("/") + "-merged", "triplets.parquet")

    merged_df.to_parquet(triplets_uri, index=False)

    print(f"Saved triplets to {triplets_uri}")
