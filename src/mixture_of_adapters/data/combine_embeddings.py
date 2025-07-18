# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
import os

import polars as pl

from tqdm import tqdm
import boto3
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

def read_parquet_file(file_key):
    s3 = boto3.client('s3')
    bucket_name = os.environ.get('BUCKET_NAME')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df_pl = pl.read_parquet(BytesIO(obj['Body'].read()))
    return df_pl.to_pandas()

def main():
    bucket_name = ""
    s3_working_dir = f"s3://{bucket_name}/embedding-adapter/data/csts/triplets-1744760875-merged/"

    s3 = boto3.client('s3')
    s3_working_dir_parts = s3_working_dir.replace("s3://", "").split("/", 1)
    bucket_name = s3_working_dir_parts[0]
    prefix = s3_working_dir_parts[1] + "embeddings/"

    os.environ['BUCKET_NAME'] = bucket_name

    paginator = s3.get_paginator('list_objects_v2')
    parquet_files = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        parquet_files.extend([
            obj['Key'] for obj in page.get('Contents', [])
            if obj['Key'].endswith(".parquet")
        ])

    with ThreadPoolExecutor(max_workers=256) as executor:
        combined_df_list = list(tqdm(executor.map(read_parquet_file, parquet_files), desc="Reading Parquet files", total=len(parquet_files)))

    combined_df = pd.concat(combined_df_list, ignore_index=True)
    print("Combined DataFrame shape:", combined_df.shape)
    
    output_uri = os.path.join(s3_working_dir, "embeddings.parquet")
    combined_df.to_parquet(output_uri, index=False)
    print(f"Combined DataFrame saved to {output_uri}")

if __name__ == "__main__":
    main()
