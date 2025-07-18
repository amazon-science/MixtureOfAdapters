# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import torch
import numpy as np
import pyarrow.dataset as ds

from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule

import os
import logging
from tqdm import tqdm


# Get logger
logger = logging.getLogger(__name__)


class CSTSDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        assert len(self.dataframe) % 2 == 0, "DataFrame length must be even for CSTS dataset. The dataset is length: {}".format(len(self.dataframe))

    def __len__(self):
        return len(self.dataframe) // 2

    def __getitem__(self, idx):
        rowA = self.dataframe.iloc[2 * idx]
        rowB = self.dataframe.iloc[2 * idx + 1]
        return {
            "sentence1": rowA['sentence1'],
            "sentence2": rowA['sentence2'],
            "conditionA": rowA['condition'],
            "conditionB": rowB['condition'],
            "embedding_sentence1": torch.tensor(rowA['sentence1_embedding'], dtype=torch.float32),
            "embedding_sentence2": torch.tensor(rowA['sentence2_embedding'], dtype=torch.float32),
            "embedding_conditionA": torch.tensor(rowA['condition_embedding'], dtype=torch.float32),
            "embedding_conditionB": torch.tensor(rowB['condition_embedding'], dtype=torch.float32),
            "label_conditionA": torch.tensor(rowA['label'], dtype=torch.long),
            "label_conditionB": torch.tensor(rowB['label'], dtype=torch.long),
        }
    

class CSTSDataModule(LightningDataModule):
    def __init__(self, local_data_path, batch_size, num_workers, use_reannotated_data=False):
        """
        Data module for the CSTS dataset.
        Args:
            local_data_path (str): Path to the local data directory.
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of workers for the DataLoader.
            task_embedding_ckpt_path (str): Path to the task embedding checkpoint.
        """
        super().__init__()
        self.local_data_path = local_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = "csts" if not use_reannotated_data else "lcsts"

    def prepare_data(self):
        """
        Downloads the Parquet files from the raw data path and saves them locally.
        """
        # Prepare data as required by the parent class
        super().prepare_data()

        # Ensure the local data path exists
        os.makedirs(self.local_data_path, exist_ok=True)

        csts_train_path = os.path.join(self.local_data_path, f"{self.dataset_name}_train.parquet")
        csts_val_path = os.path.join(self.local_data_path, f"{self.dataset_name}_val.parquet")

        # Check if the required files exist
        if all(os.path.exists(path) for path in [csts_train_path, csts_val_path]):
            logger.info("All required files exist. Skipping data preparation.")
            return
        else:
            raise FileNotFoundError(
                f"Required files not found in {self.local_data_path}. Please ensure the files are present."
            )
    
    def setup(self, stage=None):
        """
        Reads the Parquet files from the local directory into Pandas DataFrames based on the stage.
        """
        super().setup(stage=stage)
        if stage == "fit" or stage == "predict" or stage is None:
            csts_train_path = os.path.join(self.local_data_path, f"{self.dataset_name}_train.parquet")
            self.csts_train_df = pd.read_parquet(csts_train_path)
            logger.info(f"Number of CSTS training triplets: {len(self.csts_train_df)}")

        if stage == "fit" or stage == "predict" or stage == "validate" or stage is None:
            csts_val_path = os.path.join(self.local_data_path, f"{self.dataset_name}_val.parquet")
            self.csts_val_df = pd.read_parquet(csts_val_path)
            logger.info(f"Number of CSTS validation triplets: {len(self.csts_val_df)}")

    def train_dataloader(self, shuffle=True):
        csts_train_dataset = CSTSDataset(self.csts_train_df)
        csts_train_dataloader = DataLoader(csts_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)
        return csts_train_dataloader
    
    def val_dataloader(self):
        csts_val_dataset = CSTSDataset(self.csts_val_df)
        csts_val_dataloader = DataLoader(csts_val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return csts_val_dataloader
