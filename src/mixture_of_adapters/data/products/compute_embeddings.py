# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import BasePredictionWriter

from mixture_of_adapters.embedding_model import EmbeddingModel
from concurrent.futures import ThreadPoolExecutor

class EmbeddingLightningModule(LightningModule):
    def __init__(self, embedding_model: EmbeddingModel):
        super().__init__()
        self.embedding_model = embedding_model
    
    def predict_step(self, batch, batch_idx):
        embeddings = self.embedding_model.encode(batch['text'], show_progress_bar=False)

        return {
            "embedding": embeddings,
            "id": batch['id'],
        }


class WriterCallback(BasePredictionWriter):
    def __init__(self, output_dir, datamodule):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.datamodule = datamodule
        self.executor = ThreadPoolExecutor(max_workers=16)
        self.buffer = []
        self.buffer_size = 50
        self.flush_count = 0

    @staticmethod
    def save_to_s3(df: pd.DataFrame, output_uri):
        df.to_parquet(output_uri, index=False)
    
    def flush_buffer(self, global_rank):
        if not self.buffer:
            return
        
        # Combine all buffered data into a single DataFrame
        buffered_df = pd.concat(self.buffer, ignore_index=True)
        self.buffer = []  # Clear the buffer

        # Generate output URI
        output_uri = os.path.join(self.output_dir, f"embeddings_{global_rank}_{self.flush_count}.parquet")
        self.flush_count += 1

        # Submit to executor
        print(f"Flushing buffer to {output_uri} on rank {global_rank}")
        self.executor.submit(self.save_to_s3, buffered_df, output_uri)

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        embeddings = prediction["embedding"]
        ids = prediction["id"]
        
        # Convert to DataFrame
        embeddings_df = pd.DataFrame({'embedding': [row for row in embeddings.cpu().numpy()], 'id': ids})

        # Drop padding rows
        embeddings_df = embeddings_df[~embeddings_df['id'].isna()]
        if embeddings_df.empty:
            return
        
        # Add to buffer
        self.buffer.append(embeddings_df)

        # Flush buffer if it reaches the specified size
        if len(self.buffer) >= self.buffer_size:
            self.flush_buffer(trainer.global_rank)

    def on_predict_epoch_end(self, trainer, pl_module):
        # Flush remaining data in the buffer at the end of prediction
        self.flush_buffer(trainer.global_rank)


class TextDataset(Dataset):
    def __init__(self, texts_df: pd.DataFrame):
        self.texts_df = texts_df
    
    def __len__(self):
        return len(self.texts_df)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts_df.iloc[idx]["text"],
            "id": self.texts_df.iloc[idx]["id"]
        }


class TextDataModule(LightningDataModule):
    def __init__(self, s3_uri, batch_size=64):
        super().__init__()
        self.s3_uri = s3_uri
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Read texts from a Parquet file
        df = pd.read_parquet(self.s3_uri)

        # Extract texts from pairs of columns
        texts = []
        ids = []
        for column in df.columns:
            if f"id_{column}" in df.columns:
                print(f"Processing column: {column}")
                texts.extend(df[column].tolist())
                ids.extend(df[f"id_{column}"].tolist())

        # Create a DataFrame with the texts and their corresponding IDs
        self.texts_df = pd.DataFrame({"text": texts, "id": ids})
        self.texts_df = self.texts_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

        # Ensure the length of texts_df is a multiple of total_batch_size
        total_batch_size = self.batch_size * torch.distributed.get_world_size() if torch.distributed.is_initialized() else self.batch_size
        remainder = len(self.texts_df) % total_batch_size
        if remainder != 0:
            padding_size = total_batch_size - remainder
            padding_df = pd.DataFrame({
                "text": ["padding"] * padding_size,
                "id": [np.nan] * padding_size
            })
            self.texts_df = pd.concat([self.texts_df, padding_df]).reset_index(drop=True)

    def predict_dataloader(self):
        dataset = TextDataset(self.texts_df)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)
    
def main():
    """
    Generate embeddings for a list of texts using the EmbeddingModel with DataLoader.
    """
    bucket_name = ""
    s3_working_dir = f"s3://{bucket_name}/embedding-adapter/data/products/triplets-1745392949/"
    model_name = "e5"

    batch_size = 48
    datamodule = TextDataModule(s3_uri=os.path.join(s3_working_dir, "triplets.parquet"), batch_size=batch_size)
    
    # Create Lightning model
    embedding_model = EmbeddingModel(model_name=model_name)
    model = EmbeddingLightningModule(embedding_model)

    # Create WriterCallback to save predictions
    output_dir = os.path.join(s3_working_dir, f"{model_name}_embeddings")
    writer_callback = WriterCallback(output_dir=output_dir, datamodule=datamodule)

    # Create Lightning Trainer with DDP strategy
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        callbacks=[writer_callback],
    )

    # Run predictions in distributed mode
    trainer.predict(model, datamodule=datamodule, return_predictions=False)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
