# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
import json
import re
import uuid
import pandas as pd
import os
import time


from mixture_of_adapters.data.bedrock_client import BedrockClient
from mixture_of_adapters.data.utils import read_template, generate_prompts

def create_input_records(prompts, model_inputs):
    """
    Create input records for the batch job and a list with recordId and prompts.
    
    :param prompts: List of prompt strings.
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
            "prompt": prompt
        })
    return input_records, prompt_records
    

if __name__ == "__main__":
    bedrock_client = BedrockClient(region_name='ap-south-1', model_id="apac.anthropic.claude-3-5-sonnet-20241022-v2:0")

    bucket_name = ""
    output_s3_path = f"s3://{bucket_name}/embedding-adapter/data/products/tasks-{timestamp}/".format(timestamp=int(time.time()))

    # Read the template
    template_path = "/home/ec2-user/embedding-adapter/templates/products/task_template.txt"
    template = read_template(template_path)

    ATTRIBUTE_LIST = [
        "material", "color", "size", "dimensions", "weight", "capacity", "volume",
        "shape", "pattern", "style", "fit type", "neck style",
        "screen size", "storage capacity", "resolution",
        "connectivity type", "battery life", "power source", "wattage",
        "age group", "gender", "compatibility", "brand",
        "indoor/outdoor use", "temperature resistance",
        "water resistance level", "scent", "flavor", "fabric type",
        "refillable", "energy efficiency", "energy rating",
        "processor type", "form factor", "charging type", 
        "charging type", "seat height", "heat resistance",
        "price range",  "skin type compatibility", "hair type compatibility",
        "tone/shade", "formulation (e.g., cream, gel, powder)", 
        "tint", "opacity", "shade", "color family", "primary color",
        "secondary color", "design", "theme",
    ]

    print(f"Generating tasks for {len(ATTRIBUTE_LIST)} attributes.")

    variable_values = {
        "criteria": ATTRIBUTE_LIST,
    }

    # Generate prompts
    num_prompts = 5000
    prompts = generate_prompts(template, num_prompts, variable_values=variable_values)

    # print(bedrock_client.run_on_demand_inference(prompts[0]))
    # exit()

    # Convert prompts to model inputs
    model_inputs = bedrock_client.convert_prompts_to_model_inputs(prompts)

    # Create input records
    input_records, _ = create_input_records(prompts, model_inputs)

    # Submit the batch inference job
    response = bedrock_client.submit_batch_inference_job(input_records, output_s3_path)

    print("Batch inference job submitted. Response:")
    print(response)
    
    print("Output S3 path:", output_s3_path)
