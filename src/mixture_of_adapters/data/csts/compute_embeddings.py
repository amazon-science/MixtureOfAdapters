# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pyrootutils
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from mixture_of_adapters.embedding_model import EmbeddingModel


def compute_embeddings(df: pd.DataFrame, embedding_model: EmbeddingModel):
    sentence1_embeddings = embedding_model.encode(df['sentence1'].to_list(), max_length=512, batch_size=64).float().cpu().numpy()
    sentence2_embeddings = embedding_model.encode(df['sentence2'].to_list(), max_length=512, batch_size=64).float().cpu().numpy()
    condition_embeddings = embedding_model.encode(df['condition'].to_list(), max_length=512, batch_size=64).float().cpu().numpy()
    df['sentence1_embedding'] = [embedding for embedding in sentence1_embeddings]
    df['sentence2_embedding'] = [embedding for embedding in sentence2_embeddings]
    df['condition_embedding'] = [embedding for embedding in condition_embeddings]
    return df


if __name__ == "__main__":
    # Set the project root using the current working directory
    project_root = pyrootutils.setup_root(os.getcwd(), indicator=".git", pythonpath=True, cwd=True)
    print("Working directory set to:", os.getcwd())

    model_name = "simcse"

    print("Loading embedding model...")
    embedding_model = EmbeddingModel(device_map="cuda", model_name=model_name)

    train_df = pd.read_csv("raw_data/csts/csts_train.csv")
    print("Computing embeddings for training data...")
    train_df = compute_embeddings(train_df, embedding_model)
    train_df.to_parquet(f"data/csts/{model_name}/csts_train.parquet", index=False)

    val_df = pd.read_csv("raw_data/csts/csts_validation.csv")
    print("Computing embeddings for validation data...")
    val_df = compute_embeddings(val_df, embedding_model)
    val_df.to_parquet(f"data/csts/{model_name}/csts_val.parquet", index=False)

    test_df = pd.read_csv("raw_data/csts/csts_test.csv")
    print("Computing embeddings for test data...")
    test_df = compute_embeddings(test_df, embedding_model)
    test_df.to_parquet(f"data/csts/{model_name}/csts_test.parquet", index=False)

    # Relabled dataset from https://github.com/brandeis-llc/L-CSTS
    relabeled_df = pd.read_parquet(f"data/csts/{model_name}/csts_val.parquet")
    annotation_df = pd.read_csv(f"raw_data/csts/csts-val-reannotated.tsv", sep="\t")
    print("Computing embeddings for relabeled data...")
    relabeled_df['label'] = annotation_df['reannotated-label']

    train_split, val_split = relabeled_df[:-800], relabeled_df[-800:]

    train_split.to_parquet(f"data/csts/{model_name}/lcsts_train.parquet", index=False)
    val_split.to_parquet(f"data/csts/{model_name}/lcsts_val.parquet", index=False)
