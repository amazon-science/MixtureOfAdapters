# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import torch
import numpy as np
import pyarrow.dataset as ds

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from lightning.pytorch import LightningDataModule

import os
import logging
from tqdm import tqdm


# Get logger
logger = logging.getLogger(__name__)

class TripletDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            "embedding_task": torch.tensor(row['embedding_task'], dtype=torch.float32),
            "embedding_anchor": torch.tensor(row['embedding_anchor_document'], dtype=torch.float32),
            "embedding_positive": torch.tensor(row['embedding_positive_document'], dtype=torch.float32),
            "embedding_negative": torch.tensor(row['embedding_negative_document'], dtype=torch.float32),
            "task": row['task'],
            "anchor_document": row['anchor_document'],
            "positive_document": row['positive_document'],
            "negative_document": row['negative_document'],
            "id_task": row['id_task'],
            "id_anchor_document": row['id_anchor_document'],
            "id_positive_document": row['id_positive_document'],
            "id_negative_document": row['id_negative_document']
        }

class TripletDataModule(LightningDataModule):
    def __init__(self, raw_data_path=None, local_data_path="data", embedding_model="sfr", batch_size=32, num_workers=4, val_split=0.2, ood_task_split=0.5, repeat=1):
        """
        Args:
            raw_data_path (str): Raw data path. Defaults to None.
            local_data_path (str, optional): Local path to save or load data. Defaults to None.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            val_split (float): Fraction of the dataset to use for validation (between 0 and 1).
            ood_task_split (float): Fraction of the dataset to use for out-of-distribution tasks (between 0 and 1).
        """
        super().__init__()
        self.raw_data_path = raw_data_path
        self.local_data_path = local_data_path
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.ood_task_split = ood_task_split
        self.repeat = repeat

    def prepare_data(self):
        """
        Downloads the Parquet files from the raw data path and saves them locally.
        """
        # Ensure the local data path exists
        os.makedirs(os.path.join(self.local_data_path, self.embedding_model), exist_ok=True)

        train_triplets_path = os.path.join(self.local_data_path, self.embedding_model, "train_triplets.parquet")
        val_triplets_path = os.path.join(self.local_data_path, self.embedding_model, "val_triplets.parquet")
        ood_triplets_path = os.path.join(self.local_data_path, self.embedding_model, "ood_triplets.parquet")
        train_tasks_path = os.path.join(self.local_data_path, self.embedding_model, "train_tasks.parquet")
        ood_tasks_path = os.path.join(self.local_data_path, self.embedding_model, "ood_tasks.parquet")

        # Check if the required files exist
        if all(os.path.exists(path) for path in [train_triplets_path, val_triplets_path, ood_triplets_path, train_tasks_path, ood_tasks_path]):
            logger.info("All required files exist. Skipping data preparation.")
            return

        # Raw file paths
        triplets_path = os.path.join(self.raw_data_path, "triplets.parquet")
        embeddings_path = os.path.join(self.raw_data_path, "embeddings.parquet")
        if not os.path.exists(embeddings_path):
            embeddings_path = os.path.join(self.raw_data_path, f"{self.embedding_model}_embeddings/")

        logger.info("Loading raw triplets...")
        triplets = pd.read_parquet(triplets_path)
        logger.info("Triplets loaded successfully.")
        print(f"Triplets shape: {triplets.shape}")

        # Read embeddings in batches using pyarrow
        embeddings_batches = []
        dataset = ds.dataset(embeddings_path, format="parquet")
        scanner = dataset.to_batches()
        embeddings_batches = []
        for batch in tqdm(scanner, desc="Loading raw embeddings..."):
            batch_df = batch.to_pandas()
            embeddings_batches.append(batch_df[['id', 'embedding']])
        embeddings = pd.concat(embeddings_batches, ignore_index=True)
        logger.info("Embeddings loaded successfully.")
        print(f"Embeddings shape: {embeddings.shape}")

        text_columns = ["task", "anchor_document", "positive_document", "negative_document"]

        # Normalize all embeddings in the embeddings DataFrame using numpy
        embeddings["embedding"] = embeddings["embedding"].apply(
            lambda x: x / np.linalg.norm(x)
        )

        for col in text_columns:
            renamed_embeddings = embeddings.rename(columns={f"id": f"id_{col}", f"embedding": f"embedding_{col}"})
            triplets = triplets.merge(renamed_embeddings, left_on=f"id_{col}", right_on=f"id_{col}", how="inner")

        print(f"Triplets shape after merging with embeddings: {triplets.shape}")

        tasks = triplets[["task", "id_task", "embedding_task"]].drop_duplicates(subset=["id_task"]).reset_index()

        # Split the tasks into training and out-of-distribution (OOD) validation tasks
        tasks = tasks.sample(frac=1, random_state=42)
        split_idx = int(len(tasks) * (1 - self.ood_task_split))
        training_tasks = tasks[:split_idx]
        ood_validation_tasks = tasks[split_idx:]

        logger.info(f"Number of training tasks: {len(training_tasks)}")
        logger.info(f"Number of OOD validation tasks: {len(ood_validation_tasks)}")

        # Filter the triplets based on the split
        train_and_val_triplets = triplets[triplets["id_task"].isin(training_tasks['id_task'])]
        ood_triplets = triplets[triplets["id_task"].isin(ood_validation_tasks['id_task'])]

        # Shuffle the training triplets
        train_and_val_triplets = train_and_val_triplets.sample(frac=1, random_state=42)
        split_idx = int(len(train_and_val_triplets) * (1 - self.val_split))
        train_triplets = train_and_val_triplets[:split_idx]
        val_triplets = train_and_val_triplets[split_idx:]

        logger.info(f"Number of training triplets: {len(train_triplets)}")
        logger.info(f"Number of validation triplets: {len(val_triplets)}")
        logger.info(f"Number of OOD triplets: {len(ood_triplets)}")

        # Save the triplets as Parquet files in the local data path
        logger.info(f"Saving split data...")
        train_triplets.to_parquet(train_triplets_path)
        val_triplets.to_parquet(val_triplets_path)
        ood_triplets.to_parquet(ood_triplets_path)
        training_tasks.to_parquet(train_tasks_path)
        ood_validation_tasks.to_parquet(ood_tasks_path)
        logger.info(f"Split data saved at: {self.local_data_path}")

    def setup(self, stage=None):
        """
        Reads the Parquet files from the local directory into Pandas DataFrames based on the stage.
        """
        logger.info(f"Setting up data for stage: {stage}")
        if stage == "fit" or stage == "predict" or stage is None:
            train_path = os.path.join(self.local_data_path, self.embedding_model, "train_triplets.parquet")
            self.train_df = pd.read_parquet(train_path)
            logger.info(f"Number of training triplets: {len(self.train_df)}")

        if stage == "fit" or stage == "predict" or stage == "validate" or stage is None:
            val_path = os.path.join(self.local_data_path, self.embedding_model, "val_triplets.parquet")
            ood_path = os.path.join(self.local_data_path, self.embedding_model, "ood_triplets.parquet")

            self.val_df = pd.read_parquet(val_path)
            self.ood_df = pd.read_parquet(ood_path)

            logger.info(f"Number of validation triplets: {len(self.val_df)}")
            logger.info(f"Number of OOD triplets: {len(self.ood_df)}")

    def train_dataloader(self, enable_repeat=True, shuffle=True):
        train_dataset = TripletDataset(self.train_df)
        if enable_repeat and self.repeat > 1:
            logger.info(f"Repeating training dataset {self.repeat} times.")
            train_dataset = ConcatDataset([train_dataset] * self.repeat)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)
        return train_dataloader

    def id_val_dataloader(self):
        val_dataset = TripletDataset(self.val_df)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def ood_val_dataloader(self):
        ood_dataset = TripletDataset(self.ood_df)
        return DataLoader(ood_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def val_dataloader(self):
        return [self.id_val_dataloader(), self.ood_val_dataloader(), self.train_dataloader(enable_repeat=False, shuffle=False)] # Return all dataloaders for validation
    
    def predict_dataloader(self):
        return [self.train_dataloader(enable_repeat=False, shuffle=False), self.id_val_dataloader(), self.ood_val_dataloader()] # Return all dataloaders for prediction