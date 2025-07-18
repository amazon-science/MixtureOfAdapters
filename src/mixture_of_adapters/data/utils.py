# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import random
import json
import boto3


def read_template(file_path):
    """
    Read the template from a file.
    
    :param file_path: Path to the template file.
    :return: Template string.
    """
    with open(file_path, 'r') as file:
        return file.read()

def generate_prompts(template, num_prompts, variable_values: Optional[dict] = None):
    """
    Generate prompts by replacing variables in the template with random values.
    
    :param template: Template string with placeholders.
    :param variable_values: Dictionary of variable names and their possible values.
    :param num_prompts: Number of prompts to generate.
    :return: List of generated prompts. Each prompt is a dictionary with variable names as keys and their selected values.
    """
    prompts = []
    for _ in range(num_prompts):
        prompt = template
        values = {}
        if variable_values is not None:
            index = random.randint(0, len(next(iter(variable_values.values()))) - 1)
            for variable, value_list in variable_values.items():
                selected_value = value_list[index]
                prompt = prompt.replace(f"{{{{{variable}}}}}", selected_value)
                values[variable] = selected_value
        values['prompt'] = prompt
        prompts.append(values)
    return prompts

def read_jsonl_from_s3(s3_uri):
    """
    Read a JSONL file from an S3 URI and parse it into a list of dictionaries.

    :param s3_uri: S3 URI of the JSONL file (e.g., s3://bucket-name/path/to/file.jsonl).
    :return: List of dictionaries parsed from the JSONL file.
    """
    s3 = boto3.client('s3')
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    return [json.loads(line) for line in content.splitlines() if line.strip()]