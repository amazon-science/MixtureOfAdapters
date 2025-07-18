# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import torch
import numpy as np
import pyarrow.dataset as ds

from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
from mixture_of_adapters.task_triplet_module import TaskTripletModule

import os
import logging
from tqdm import tqdm


# Get logger
logger = logging.getLogger(__name__)


class KGCDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            # "head_embedding": torch.tensor(row['head_embedding'], dtype=torch.float32),
            # "tail_embedding": torch.tensor(row['tail_embedding'], dtype=torch.float32),
            # "relation_embedding": torch.tensor(row['relation_embedding'], dtype=torch.float32),
            "head_desc": row['head_desc'],
            "tail_desc": row['tail_desc'],
            "relation": row['relation'],
        }
    

class KGCDataModule(LightningDataModule):
    def __init__(self, local_data_path, batch_size, num_workers, task_embedding_ckpt_path=None, use_reannotated_data=False):
        """
        Data module for the KGC dataset.
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
        if task_embedding_ckpt_path is not None:
            task_triplet_module = TaskTripletModule.load_from_checkpoint(task_embedding_ckpt_path)
            task_triplet_module.eval()
            self.task_embedding_transformer = task_triplet_module.adapter
        else:
            self.task_embedding_transformer = None

    def prepare_data(self):
        """
        Downloads the Parquet files from the raw data path and saves them locally.
        """
        # Prepare data as required by the parent class
        super().prepare_data()

        # Ensure the local data path exists
        os.makedirs(self.local_data_path, exist_ok=True)

        kgc_train_path = os.path.join(self.local_data_path, "kgc_train.parquet")
        kgc_val_path = os.path.join(self.local_data_path, "kgc_val.parquet")

        # Check if the required files exist
        if all(os.path.exists(path) for path in [kgc_train_path, kgc_val_path]):
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
            kgc_train_path = os.path.join(self.local_data_path, "kgc_train.parquet")
            self.kgc_train_df = pd.read_parquet(kgc_train_path)
            logger.info(f"Number of KGC training triplets: {len(self.kgc_train_df)}")

        if stage == "fit" or stage == "predict" or stage == "validate" or stage is None:
            kgc_val_path = os.path.join(self.local_data_path, "kgc_val.parquet")
            self.kgc_val_df = pd.read_parquet(kgc_val_path)
            logger.info(f"Number of KGC validation triplets: {len(self.kgc_val_df)}")

    def train_dataloader(self, shuffle=True):
        kgc_train_dataset = KGCDataset(self.kgc_train_df)
        kgc_train_dataloader = DataLoader(kgc_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)
        return kgc_train_dataloader
    
    def val_dataloader(self):
        kgc_val_dataset = KGCDataset(self.kgc_val_df)
        kgc_val_dataloader = DataLoader(kgc_val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return kgc_val_dataloader
