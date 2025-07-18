# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from lightning.pytorch.core.module import LightningModule
from embedding_adapter.products_module import TripletModule

from embedding_adapter.embedding_model import EmbeddingModel
from embedding_adapter.losses import InBatchInfoNCELoss
from embedding_adapter.moa import MixtureOfAdapters  

from scipy.stats import spearmanr
from hydra.utils import instantiate

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadNetwork(nn.Module):
    def __init__(self, input_size, context_size, adapted_size, hidden_size, noise_sd=0.01):
        super(HeadNetwork, self).__init__()

        self.noise_sd = noise_sd

        self.context_transformer = nn.Sequential(
            nn.Linear(context_size, input_size),
        )

        concat_size = input_size + adapted_size
        self.final_network = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(concat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def apply_noise(self, tensor):
        noise = self.noise_sd * torch.randn_like(tensor)
        return tensor + noise
        
    def forward(self, document_emb, context_emb, adapted_emb, use_noise=False):
        if use_noise:
            document_emb = self.apply_noise(document_emb)
            context_emb = self.apply_noise(context_emb)
            adapted_emb = self.apply_noise(adapted_emb)
        
        context_emb = self.context_transformer(context_emb)

        document_emb = document_emb * context_emb

        final_input = torch.cat([document_emb, adapted_emb], dim=1)

        return self.final_network(final_input)

class KGCModule(LightningModule):
    def __init__(self, adapter=None, adapter_ckpt_path=None, learning_rate=1e-3, weight_decay=1e-2,
                 contrastive_loss_coeff=1,
                 load_loss_coefficient=0, consistency_loss_coefficient=0,
                 initial_tau=1, min_tau=0.1, tau_anneal_rate=0.1,
                 additive_margin=0):
        """
        A simple triplet model for embedding learning.
        Args:
            adapter (dict): Configuration for the adapter.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            transform (bool): Whether to apply a transformation to the output of the adapter.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.contrastive_loss_coeff = contrastive_loss_coeff
        self.load_loss_coefficient = load_loss_coefficient
        self.consistency_loss_coefficient = consistency_loss_coefficient

        # Temperature annealing parameters
        self.tau = initial_tau
        self.min_tau = min_tau
        self.tau_anneal_multiplier = np.exp(-tau_anneal_rate)
        
        self.save_hyperparameters()

        self.encoder = EmbeddingModel(model_name="bert", torch_dtype=torch.float32, device_map="cuda")

        # Instantiate the adapter
        if adapter_ckpt_path is None:
            self.adapter = instantiate(adapter)
            self.adapter: MixtureOfAdapters = instantiate(adapter)
        else:
            # self.adapter = CSTSQuadModule.load_from_checkpoint(adapter_ckpt_path).adapter
            self.adapter = TripletModule.load_from_checkpoint(adapter_ckpt_path).adapter

        # Loss function
        self.contrastive_loss_fn = InBatchInfoNCELoss(tau=0.1, additive_margin=additive_margin)

        # Train epoch storage
        self.train_epoch_scores = []
        self.train_epoch_labels = []
    
    def log_metric(self, loss, name, split, batch_size):
        self.log(f"{name}/{split}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, add_dataloader_idx=False)

    def forward(self, batch, split):
        inputs = self.encoder.tokenize(batch["head_desc"], max_length=128).to(self.encoder.model.device)
        head_embedding = self.encoder(**inputs)

        inputs = self.encoder.tokenize(batch["tail_desc"], max_length=128).to(self.encoder.model.device)
        tail_embedding = self.encoder(**inputs)

        inputs = self.encoder.tokenize(batch["relation"], max_length=128).to(self.encoder.model.device)
        relation_embedding = self.encoder(**inputs)

        batch_size = len(head_embedding)

        # Compute encodings
        gates, load_loss, consistency_loss = self.adapter.forward_gate(relation_embedding)
        head_embedding_adapted, _ = self.adapter.forward_adapter(head_embedding, gates, return_expert_outputs=True)

        # Contrastive_loss loss
        contrastive_loss = self.contrastive_loss_fn(head_embedding_adapted, tail_embedding)

        # Add losses
        loss = 0
        loss += self.contrastive_loss_coeff * contrastive_loss
        loss += self.load_loss_coefficient * load_loss
        loss += self.consistency_loss_coefficient * consistency_loss

        self.log_metric(contrastive_loss, "contrastive_loss", split, batch_size)
        self.log_metric(load_loss, "load_loss", split, batch_size)
        self.log_metric(consistency_loss, "consistency_loss", split, batch_size)
        self.log_metric(loss, "total_loss", split, batch_size) 

        return loss

    def training_step(self, batch, batch_idx):     
        return self(batch, split="train")
    
    def validation_step(self, batch, batch_idx):
        return self(batch, split="val")

    def configure_optimizers(self):
        adapter_params = [param for name, param in self.adapter.named_parameters() if param.requires_grad]
        other_params = [param for name, param in self.named_parameters() if not name.startswith("adapter")]

        optimizer = torch.optim.AdamW([
            {"params": adapter_params, "lr": self.learning_rate * 0.01},  # Smaller learning rate for adapter
            {"params": other_params, "lr": self.learning_rate}
        ], weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        """Set the temperature parameter at the start of each training epoch"""
        self.adapter.set_tau(self.tau)
        self.log("temperature", self.tau, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.tau = max(self.tau * self.tau_anneal_multiplier, self.min_tau)
