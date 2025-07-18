# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import numpy as np

import torch
from lightning.pytorch.core.module import LightningModule
from mixture_of_adapters.moa import MixtureOfAdapters

from mixture_of_adapters.losses import TripletLoss, JointInfoNCELoss
from hydra.utils import instantiate

logger = logging.getLogger(__name__)

class TripletModule(LightningModule):
    def __init__(
            self, 
            adapter, 
            learning_rate=1e-3, 
            weight_decay=1e-2,
            initial_tau=1.0, 
            min_tau=0.01, 
            tau_anneal_rate=0.01,
            triplet_loss_coefficient=1.0, 
            loss_temperature=0.3,
            in_batch_loss_coefficient=0, 
            in_batch_loss_temperature=0.1,
            similarity="cosine", 
        ):
        """
        A simple triplet model for embedding learning.
        Args:
            adapter : Adapter object to be used for embedding adaptation.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            initial_tau (float): Initial temperature for the gating mechanism.
            min_tau (float): Minimum temperature for the gating mechanism.
            tau_anneal_rate (float): Rate of temperature annealing.
            triplet_loss_coefficient (float): Coefficient for the triplet loss.
            loss_temperature (float): Temperature for the triplet loss.
            in_batch_loss_coefficient (float): Coefficient for the in-batch loss.
            in_batch_loss_temperature (float): Temperature for the in-batch loss.
            similarity (str): Similarity metric to be used. Default is "cosine".
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.triplet_loss_coefficient = triplet_loss_coefficient
        self.in_batch_loss_coefficient = in_batch_loss_coefficient
        
        # Temperature annealing parameters
        self.tau = initial_tau
        self.min_tau = min_tau
        self.tau_anneal_multiplier = np.exp(-tau_anneal_rate)
        
        self.save_hyperparameters()

        # Instantiate the adapter
        if 'moa' in adapter['_target_']:
            self.adapter: MixtureOfAdapters = instantiate(adapter)
            self.moe_adapter = True
        else:
            self.adapter = instantiate(adapter)
            self.moe_adapter = False

        # Loss function
        self.loss_fn = TripletLoss(tau=loss_temperature, similarity=similarity)  
        self.in_batch_loss_fn = JointInfoNCELoss(tau=in_batch_loss_temperature, similarity=similarity)   

    def encode(self, embedding_task, embedding_anchor, embedding_positive, embedding_negative):
        if self.moe_adapter:
            gates = self.adapter.forward_gate(embedding_task)
            anchor_emb = self.adapter.forward_adapter(embedding_anchor, gates)
            positive_emb = self.adapter.forward_adapter(embedding_positive, gates)
            negative_emb = self.adapter.forward_adapter(embedding_negative, gates)
        else:
            anchor_emb = self.adapter(embedding_anchor, embedding_task)
            positive_emb = self.adapter(embedding_positive, embedding_task)
            negative_emb = self.adapter(embedding_negative, embedding_task)
        return anchor_emb, positive_emb, negative_emb
    
    def compute_losses(self, embedding_task, embedding_anchor, embedding_positive, embedding_negative):
        if self.moe_adapter:
            gates = self.adapter.forward_gate(embedding_task)
            anchor_emb, anchor_expert_outputs = self.adapter.forward_adapter(embedding_anchor, gates, return_expert_outputs=True)
            positive_emb, positive_expert_outputs = self.adapter.forward_adapter(embedding_positive, gates, return_expert_outputs=True)
            negative_emb, negative_expert_outputs = self.adapter.forward_adapter(embedding_negative, gates, return_expert_outputs=True)

            triplet_loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)
            if self.in_batch_loss_coefficient > 0:
                in_batch_loss = self.in_batch_loss_fn(anchor_expert_outputs, positive_expert_outputs, negative_expert_outputs, gates)
            else:
                in_batch_loss = 0

            return triplet_loss, in_batch_loss
        else:
            anchor_emb = self.adapter(embedding_anchor, embedding_task)
            positive_emb = self.adapter(embedding_positive, embedding_task)
            negative_emb = self.adapter(embedding_negative, embedding_task)

            triplet_loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)

            return triplet_loss, 0
    
    def forward(self, batch, split):
        task = batch["embedding_task"]
        anchor = batch["embedding_anchor"]
        positive = batch["embedding_positive"]
        negative = batch["embedding_negative"]

        # Compute loss
        batch_size = len(anchor)
        triplet_loss, in_batch_loss = self.compute_losses(task, anchor, positive, negative)

        # Add losses
        loss = 0
        loss += self.triplet_loss_coefficient * triplet_loss
        loss += self.in_batch_loss_coefficient * in_batch_loss 

        self.log_metric(triplet_loss, "triplet_loss", split, batch_size) 
        self.log_metric(in_batch_loss, "in_batch_loss", split, batch_size)
        self.log_metric(loss, "total_loss", split, batch_size) 

        return loss

    def log_metric(self, loss, name, split, batch_size):
        self.log(f"{name}/{split}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, add_dataloader_idx=False)

    def training_step(self, batch, batch_idx):
        return self(batch, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx):
        # Log loss based on dataloader index
        if dataloader_idx == 0:
            split = "id_val"
        elif dataloader_idx == 1:
            split = "ood_val"
        elif dataloader_idx == 2:
            split = f"train_eval"
        else:
            split = "unknown"
            raise ValueError(f"Unknown dataloader index: {dataloader_idx}")   

        return self(batch, split)

    def predict_step(self, batch: dict, batch_idx, dataloader_idx):
        task = batch["embedding_task"]
        anchor = batch["embedding_anchor"]
        positive = batch["embedding_positive"]
        negative = batch["embedding_negative"]

        anchor_emb, positive_emb, negative_emb = self.encode(task, anchor, positive, negative)

        batch.update({
            "adapted_embedding_anchor": anchor_emb,
            "adapted_embedding_positive": positive_emb,
            "adapted_embedding_negative": negative_emb
        })

        return batch
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
    def on_train_epoch_start(self):
        """Set the temperature parameter at the start of each training epoch"""
        if self.moe_adapter:
            self.adapter.set_tau(self.tau)
            self.log("temperature", self.tau, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.tau = max(self.tau * self.tau_anneal_multiplier, self.min_tau)
