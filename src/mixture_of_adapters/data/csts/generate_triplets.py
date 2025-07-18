# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
import pandas as pd
import os

from datasets import load_dataset

from mixture_of_adapters.data.bedrock_client import BedrockClient
from mixture_of_adapters.data.utils import read_template, generate_prompts

def create_input_records(prompts, model_inputs):
    """
    Create input records for the batch job and a list with recordId and prompts.
    
    :param prompts: List of generated prompts.
    :param model_inputs: List of model input dictionaries.
    :return: Tuple containing:
             - List of input records for the batch job.
             - List of dictionaries with recordId and prompts.
    """
    input_records = []
    prompt_records = []
    for i, (prompt, model_input) in enumerate(zip(prompts, model_inputs)):
        record_id = f"CALL{i:07d}"
        input_records.append(json.dumps({
            "recordId": record_id,
            "modelInput": model_input
        }))
        prompt_records.append({
            "recordId": record_id,
            "condition": prompt['condition']
        })
    return input_records, prompt_records    

if __name__ == "__main__":
    bedrock_client = BedrockClient(region_name='ap-south-1', model_id="apac.anthropic.claude-3-5-sonnet-20241022-v2:0")

    bukcet_name = ""
    output_s3_path = f"s3://{bukcet_name}/embedding-adapter/data/csts/triplets-{int(time.time())}/"

    # Read the template
    template_path = "templates/csts/triplet_template_claude.txt"
    template = read_template(template_path)

    # Read the raw dataset
    csts_dataset = pd.read_csv("raw_data/csts/csts_train.csv")

    conditions = csts_dataset['condition'].tolist() + csts_dataset['condition'].tolist()
    sentences = csts_dataset['sentence1'].tolist() + csts_dataset['sentence2'].tolist()

    variable_values = {
        "condition": conditions,
        "exampleSentence": sentences,
    }

    # Generate prompts
    num_prompts = 50_000
    prompts = generate_prompts(template, num_prompts, variable_values=variable_values)

    # Convert prompts to model inputs
    model_inputs = bedrock_client.convert_prompts_to_model_inputs(prompts)

    # Create input records
    input_records, prompt_records = create_input_records(prompts, model_inputs)

    # save the tasks as pandas parquet file 
    prompt_records_s3_uri = os.path.join(output_s3_path, "prompt_records.parquet")
    pd.DataFrame(prompt_records).to_parquet(prompt_records_s3_uri, index=False)

    # Submit the batch inference job
    response = bedrock_client.submit_batch_inference_job(input_records, output_s3_path)

    print("Batch inference job submitted. Response:")
    print(response)

    print("Prompt records saved to:", prompt_records_s3_uri)
    print("Output S3 path:", output_s3_path)
