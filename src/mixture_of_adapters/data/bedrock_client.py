# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import boto3
import time
import os
from urllib.parse import urlparse
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from retry import retry


def s3_uri_to_bucket_and_prefix(s3_uri):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')  # Remove leading slash
    return bucket, prefix


class BedrockClient:
    def __init__(self, region_name, model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"):
        """
        Initialize the BedrockClient with AWS credentials and region.
        """
        self.region_name = region_name
        self.client = boto3.client(
            'bedrock',
            region_name=self.region_name
        )
        self.brt = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name
        )
        self.model_id = model_id

    def convert_prompts_to_model_inputs(self, prompts):
        """
        Convert a list of prompts into model inputs.

        :param prompts: List of prompt strings.
        :return: List of model input dictionaries.
        """
        return [self.get_model_input(prompt['prompt']) for prompt in prompts]

    def save_input_to_s3(self, input_records, folder_s3_path, batch_size=100):
        """
        Save the input records to an S3 bucket.
        
        :param input_records: List of input records to save.
        :param folder_s3_path: S3 path where the records will be saved.
        :return: S3 URI of the saved file.
        """
        try:
            s3 = boto3.client('s3')
            bucket_name, folder_name = s3_uri_to_bucket_and_prefix(folder_s3_path)
            file_name = os.path.join(folder_name, 'input_prompts.jsonl')
            s3.put_object(
                Bucket=bucket_name,
                Key=file_name,
                Body='\n'.join(input_records)
            )
            return f"s3://{bucket_name}/{file_name}"
        except Exception as e:
            print(f"Error saving to S3: {e}")
            raise

    def get_model_input(self, prompt):
        """
        Convert a prompt into a model input format.

        :param prompt: The input prompt string.
        :return: Model input dictionary.
        """
        if 'claude' in self.model_id:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        elif 'llama' in self.model_id:
            return {
                "prompt": f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

                {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                "temperature": 1.0,
                "max_gen_len": 1024
            }
        else:
            raise ValueError(f"Unsupported model ID: {self.model_id}. Please provide a valid model ID.")
        
    def run_on_demand_inference(self, prompt):
        """
        Run an on-demand inference job with a single prompt.

        :param prompt: The input prompt string.
        :param model_id: ID of the model to use for inference.
        :param role_arn: IAM role ARN with permissions for the job.
        :return: Response from the Bedrock service.
        """
        try:
            # Prepare the model input
            model_input = self.get_model_input(prompt['prompt'])

            # Invoke the model directly
            response = self.brt.invoke_model(
                modelId=self.model_id,
                body=json.dumps(model_input)
            )

            # Decode the response body.
            model_response = json.loads(response["body"].read())

            # Extract the response text.
            return model_response["content"][0]["text"]
        
        except Exception as e:
            print(f"Error running on-demand inference: {e}")
            raise

    def run_batch_on_demand_inference(self, prompts):
        """
        Run a batch inference job with multiple prompts.

        :param input_records: List of input records to process.
        :param folder_s3_path: S3 path where the records will be saved.
        :return: Response from the Bedrock service.
        """
        try:
            # Run on-demand inference for each prompt
            @retry(tries=5, delay=30, max_delay=120, backoff=2)
            def process_prompt(prompt):
                return self.run_on_demand_inference(prompt)

            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(tqdm(executor.map(process_prompt, prompts), total=len(prompts), desc="Inference"))
            return responses
        
        except Exception as e:
            print(f"Error running batch on-demand inference: {e}")
            raise

    def submit_batch_inference_job(self, input_records, folder_s3_path):
        """
        Submit a batch inference job to Bedrock.
        
        :param job_name: Name of the batch job.
        :param model_id: ID of the model to use for inference.
        :param input_data_s3_uri: S3 URI for the input data.
        :param output_data_s3_uri: S3 URI for the output data.
        :param role_arn: IAM role ARN with permissions for the job.
        :return: Response from the Bedrock service.
        """
        try:
            input_s3_uri = self.save_input_to_s3(input_records, folder_s3_path)

            inputDataConfig=({
                "s3InputDataConfig": {
                    "s3Uri": input_s3_uri
                }
            })

            outputDataConfig=({
                "s3OutputDataConfig": {
                    "s3Uri": folder_s3_path
                }
            })

            jobName = f"embedding-adapter-{int(time.time())}"

            if self.region_name == 'us-east-1':
                roleArn = "arn:aws:iam::959435299807:role/service-role/EmbeddingAdapterBedrock-east-1"
            elif self.region_name == 'ap-south-1':
                roleArn = "arn:aws:iam::959435299807:role/service-role/EmbeddingAdapterBedrock-ap-south-1"

            response = self.client.create_model_invocation_job(
                roleArn=roleArn,
                modelId=self.model_id,
                jobName=jobName,
                inputDataConfig=inputDataConfig,
                outputDataConfig=outputDataConfig,
                timeoutDurationInHours = 7 * 24, # 7 days
            )

            return response
        
        except Exception as e:
            print(f"Error submitting batch inference job: {e}")
            raise

    def get_batch_inference_job_status(self, job_id):
        """
        Get the status of a batch inference job.
        
        :param job_id: ID of the batch job.
        :return: Status of the job.
        """
        try:
            response = self.client.describe_batch_inference_job(JobId=job_id)
            return response.get('Status', 'Unknown')
        except Exception as e:
            print(f"Error fetching job status: {e}")
            raise

if __name__ == "__main__":
    bedrock_client = BedrockClient(region_name='us-east-1')
    bedrock_client.run_batch_on_demand_inference(["test", "test2"])