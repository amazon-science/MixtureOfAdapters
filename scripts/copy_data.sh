#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# This script copies the triplet data and embedding data from S3 to local storage.
# Usage for products data: `bash scripts/copy_data.sh products triplets-1745392949 sfr`

BUCKET_NAME="$1"
DATA_TYPE="$2"
JOB_NAME="$2"
EMBEDDING_MODEL="$3"

TRIPLET_S3_PATH="s3://${BUCKET_NAME}/embedding-adapter/data/${DATA_TYPE}/${JOB_NAME}/triplets.parquet"
EMBEDDING_S3_PATH="s3://${BUCKET_NAME}/embedding-adapter/data/${DATA_TYPE}/${JOB_NAME}/${EMBEDDING_MODEL}_embeddings/"

aws s3 cp "$TRIPLET_S3_PATH" "./raw_data/${DATA_TYPE}/triplets.parquet"
aws s3 cp --recursive "$EMBEDDING_S3_PATH" "./raw_data/${DATA_TYPE}/${EMBEDDING_MODEL}_embeddings/"
