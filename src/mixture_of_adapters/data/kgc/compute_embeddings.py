# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pyrootutils
import os

import pandas as pd
import json

from embedding_adapter.embedding_model import EmbeddingModel


def compute_embeddings(df: pd.DataFrame, embedding_model: EmbeddingModel):
    head_embeddings = embedding_model.encode(df['head_desc'].to_list(), max_length=512, batch_size=64).float().cpu().numpy()
    tail_embeddings = embedding_model.encode(df['tail_desc'].to_list(), max_length=512, batch_size=64).float().cpu().numpy()
    relation_embeddings = embedding_model.encode(df['relation'].to_list(), max_length=512, batch_size=64).float().cpu().numpy()
    df['head_embedding'] = [embedding for embedding in head_embeddings]
    df['tail_embedding'] = [embedding for embedding in tail_embeddings]
    df['relation_embedding'] = [embedding for embedding in relation_embeddings]
    return df

def join_descriptions(df: pd.DataFrame, entities_df: pd.DataFrame):
    # Perform the join for head entities
    merged_df = df.merge(entities_df, left_on='head_id', right_on='entity_id', how='left')
    merged_df.rename(columns={'entity_desc': 'head_desc'}, inplace=True)
    merged_df.drop(columns=['entity_id', 'entity'], inplace=True)

    # Perform the join for tail entities
    merged_df = merged_df.merge(entities_df, left_on='tail_id', right_on='entity_id', how='left')
    merged_df.rename(columns={'entity_desc': 'tail_desc'}, inplace=True)
    merged_df.drop(columns=['entity_id', 'entity'], inplace=True)

    return merged_df


if __name__ == "__main__":
    # Set the project root using the current working directory
    project_root = pyrootutils.setup_root(os.getcwd(), indicator=".git", pythonpath=True, cwd=True)
    print("Working directory set to:", os.getcwd())

    model_name = "bert"

    # Create save directory if it doesn't exist
    os.makedirs(f"data/kgc/{model_name}", exist_ok=True)

    print("Loading embedding model...")
    embedding_model = EmbeddingModel(device_map="cuda", model_name=model_name)

    entitites = json.load(open('raw_data/kgc/WN18RR/entities.json', 'r', encoding='utf-8'))
    entities_df = pd.DataFrame(entitites)

    print("Computing embeddings for training data...")
    train_data = json.load(open('raw_data/kgc/WN18RR/train.txt.json', 'r', encoding='utf-8'))
    train_df = pd.DataFrame.from_dict(train_data)
    train_df = join_descriptions(train_df, entities_df)
    train_df.to_parquet(f"data/kgc/kgc_train.parquet", index=False)
    train_df = compute_embeddings(train_df, embedding_model)
    train_df.to_parquet(f"data/kgc/{model_name}/kgc_train.parquet", index=False)

    print("Computing embeddings for validation data...")
    valid_data = json.load(open('raw_data/kgc/WN18RR/valid.txt.json', 'r', encoding='utf-8'))
    valid_df = pd.DataFrame.from_dict(valid_data)
    valid_df = join_descriptions(valid_df, entities_df)
    valid_df.to_parquet(f"data/kgc/kgc_val.parquet", index=False)
    valid_df = compute_embeddings(valid_df, embedding_model)
    valid_df.to_parquet(f"data/kgc/{model_name}/kgc_val.parquet", index=False)

    print("Computing embeddings for test data...")
    test_data = json.load(open('raw_data/kgc/WN18RR/test.txt.json', 'r', encoding='utf-8'))
    test_df = pd.DataFrame.from_dict(test_data)
    test_df = join_descriptions(test_df, entities_df)
    test_df.to_parquet(f"data/kgc/kgc_test.parquet", index=False)
    test_df = compute_embeddings(test_df, embedding_model)
    test_df.to_parquet(f"data/kgc/{model_name}/kgc_test.parquet", index=False)
