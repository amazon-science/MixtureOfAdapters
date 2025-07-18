# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from lightning.pytorch.core.module import LightningModule
from mixture_of_adapters.products_module import TripletModule

from mixture_of_adapters.losses import LabeledInfoNCELoss
from mixture_of_adapters.moa import MixtureOfAdapters

from scipy.stats import spearmanr
from hydra.utils import instantiate

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F


def quad_loss_fn(score1, score2, label1, label2, margin=0.5):
    target = torch.sign(label1 - label2).detach()
    gap = score1 - score2
    return torch.mean(F.relu(margin - target * gap))

class HeadNetwork(nn.Module):
    def __init__(self, adapted_size, output_size):
        super(HeadNetwork, self).__init__()

        self.final_network = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(adapted_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        
    def forward(self, adapted_emb):
        return self.final_network(adapted_emb)

class CSTSQuadModule(LightningModule):
    def __init__(self, 
                 adapter=None, 
                 adapter_ckpt_path=None, 
                 learning_rate=1e-3, 
                 weight_decay=1e-2,
                 contrastive_loss_coeff=1, 
                 mse_loss_coeff=1, 
                 info_nce_loss_coeff=1,
                 initial_tau=1, 
                 min_tau=0.1, 
                 tau_anneal_rate=0.1, 
                 use_head=True,
                 finetune_encoder=False,
        ):
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
        self.mse_loss_coeff = mse_loss_coeff
        self.info_nce_loss_coeff = info_nce_loss_coeff

        # Temperature annealing parameters
        self.tau = initial_tau
        self.min_tau = min_tau
        self.tau_anneal_multiplier = np.exp(-tau_anneal_rate)

        self.use_head = use_head
        self.finetune_encoder = finetune_encoder
        
        self.save_hyperparameters()

        if self.finetune_encoder:
            self.encoder = EmbeddingModel(model_name="simcse", torch_dtype=torch.float32, device_map="cuda")

        # Instantiate the adapter
        if adapter_ckpt_path is None:
            self.adapter = instantiate(adapter)
            self.adapter: MixtureOfAdapters = instantiate(adapter)
        else:
            self.adapter = CSTSQuadModule.load_from_checkpoint(adapter_ckpt_path).adapter
            # self.adapter = TripletModule.load_from_checkpoint(adapter_ckpt_path).adapter
            
        if self.use_head:
            self.head_network = HeadNetwork(
                adapted_size=self.adapter.output_size, 
                output_size=self.adapter.output_size,
            )

        # Loss function
        self.mse_loss_fn = nn.MSELoss()
        self.contrastive_loss_fn = quad_loss_fn
        self.info_nce_loss_fn = LabeledInfoNCELoss(tau=0.1)

        # Validation epoch storage
        self.validation_epoch_scores = []
        self.validation_epoch_labels = []
        # Train epoch storage
        self.train_epoch_scores = []
        self.train_epoch_labels = []
    
    def log_metric(self, loss, name, split, batch_size):
        self.log(f"{name}/{split}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size, add_dataloader_idx=False)

    def forward(self, batch, split):
        if self.finetune_encoder:
            inputs = self.encoder.tokenize(batch["sentence1"], max_length=128).to(self.encoder.model.device)
            sentence1 = self.encoder(**inputs)

            inputs = self.encoder.tokenize(batch["sentence2"], max_length=128).to(self.encoder.model.device)
            sentence2 = self.encoder(**inputs)

            inputs = self.encoder.tokenize(batch["conditionA"], max_length=128).to(self.encoder.model.device)
            conditionA = self.encoder(**inputs)

            inputs = self.encoder.tokenize(batch["conditionB"], max_length=128).to(self.encoder.model.device)
            conditionB = self.encoder(**inputs)
        else:
            sentence1 = batch["embedding_sentence1"]
            sentence2 = batch["embedding_sentence2"]
            conditionA = batch["embedding_conditionA"]
            conditionB = batch["embedding_conditionB"]

        labelA = (batch["label_conditionA"].unsqueeze(-1) - 1) / 4
        labelB = (batch["label_conditionB"].unsqueeze(-1) - 1) / 4

        batch_size = len(sentence1)

        # Compute encodings
        gatesA = self.adapter.forward_gate(conditionA)
        gatesB = self.adapter.forward_gate(conditionB)

        sentence1_conditionA, sentence1_experts = self.adapter.forward_adapter(sentence1, gatesA, return_expert_outputs=True)
        sentence1_conditionB, _ = self.adapter.forward_adapter(sentence1, gatesB, return_expert_outputs=True)
        sentence2_conditionA, sentence2_experts = self.adapter.forward_adapter(sentence2, gatesA, return_expert_outputs=True)
        sentence2_conditionB, _ = self.adapter.forward_adapter(sentence2, gatesB, return_expert_outputs=True)

        conditionA_similarity = F.cosine_similarity(sentence1_conditionA, sentence2_conditionA, dim=-1).unsqueeze(-1)
        conditionB_similarity = F.cosine_similarity(sentence1_conditionB, sentence2_conditionB, dim=-1).unsqueeze(-1)

        if self.use_head:
            scoring_vector_1A = self.head_network(sentence1_conditionA)
            scoring_vector_1B = self.head_network(sentence1_conditionB)
            scoring_vector_2A = self.head_network(sentence2_conditionA)
            scoring_vector_2B = self.head_network(sentence2_conditionB)
            conditionA_score = torch.sum(scoring_vector_1A * scoring_vector_2A, dim=-1).unsqueeze(-1)
            conditionB_score = torch.sum(scoring_vector_1B * scoring_vector_2B, dim=-1).unsqueeze(-1)
        else:
            conditionA_score = conditionA_similarity
            conditionB_score = conditionB_similarity

        # Compute contrastive loss
        contrastive_loss = self.contrastive_loss_fn(conditionA_similarity, conditionB_similarity, labelA, labelB)

        # MSE loss
        all_scores = torch.cat([conditionA_score, conditionB_score], dim=0)
        all_labels = torch.cat([labelA, labelB], dim=0)
        mse_loss = self.mse_loss_fn(all_scores, all_labels.float())

        if split == "val":
            self.validation_epoch_scores.append(all_scores)
            self.validation_epoch_labels.append(all_labels)
        elif split == "train":
            self.train_epoch_scores.append(all_scores)
            self.train_epoch_labels.append(all_labels)

        # InfoNCE loss
        stacked_sentence1_experts = torch.cat([sentence1_experts, sentence1_experts], dim=0)
        stacked_sentence2_experts = torch.cat([sentence2_experts, sentence2_experts], dim=0)
        stacked_gates = torch.cat([gatesA, gatesB], dim=0)
        positiveA = labelA > labelB
        positiveB = labelB > labelA
        stacked_labels = torch.cat([positiveA, positiveB], dim=0).squeeze()
        info_nce_loss = self.info_nce_loss_fn(stacked_sentence1_experts, stacked_sentence2_experts, stacked_gates, stacked_labels)

        # Accuracy
        acc_scores = torch.sign(conditionA_score - conditionB_score) * torch.sign(labelA - labelB)
        accuracy = (acc_scores >= 0).float().sum() / len(acc_scores)

        # Add losses
        loss = 0
        loss += self.contrastive_loss_coeff * contrastive_loss
        loss += self.mse_loss_coeff * mse_loss 
        loss += self.info_nce_loss_coeff * info_nce_loss

        self.log_metric(contrastive_loss, "contrastive_loss", split, batch_size) 
        self.log_metric(mse_loss, "mse_loss", split, batch_size) 
        self.log_metric(info_nce_loss, "info_nce_loss", split, batch_size)
        self.log_metric(loss, "total_loss", split, batch_size) 

        self.log_metric(accuracy, "accuracy", split, batch_size)

        return loss

    def training_step(self, batch, batch_idx):     
        return self(batch, split="train")
    
    def validation_step(self, batch, batch_idx):
        return self(batch, split="val")

    def on_train_epoch_end(self):
        all_scores = torch.cat(self.train_epoch_scores, dim=0).cpu().detach().numpy()
        all_labels = torch.cat(self.train_epoch_labels, dim=0).cpu().detach().numpy()

        spearman_corr = spearmanr(all_scores, all_labels).correlation
        self.log("spearman_corr/train", spearman_corr, on_epoch=True, prog_bar=True, logger=True)

        # Clear the lists for the next epoch
        self.train_epoch_scores = []
        self.train_epoch_labels = []

    def on_validation_epoch_end(self):
        all_scores = torch.cat(self.validation_epoch_scores, dim=0).cpu().detach().numpy()
        all_labels = torch.cat(self.validation_epoch_labels, dim=0).cpu().detach().numpy()

        spearman_corr = spearmanr(all_scores, all_labels).correlation
        self.log("spearman_corr/val", spearman_corr, on_epoch=True, prog_bar=True, logger=True)

        # Clear the lists for the next epoch
        self.validation_epoch_scores = []
        self.validation_epoch_labels = []

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
