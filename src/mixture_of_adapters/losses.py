# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityComputer(nn.Module):
    def __init__(self, similarity_metric: str = 'cosine', tau: float = 1.0):
        """
        A utility class to compute similarity scores between two sets of embeddings.
        Args:
            similarity_metric (str): The similarity metric to use ('cosine' or 'l2').
            tau (float): Temperature scaling factor.
        """
        super().__init__()
        self.similarity_metric = similarity_metric
        self.tau = tau

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between two sets of embeddings.
        Args:
            emb1 (torch.Tensor): Embeddings of the first set (*batch_size, embedding_dim).
            emb2 (torch.Tensor): Embeddings of the second set (*batch_size, embedding_dim).
        Returns:
            torch.Tensor: Computed similarity scores of shape (*batch_size).
        """
        if self.similarity_metric == 'cosine':
            if emb1.dim() == 2:
                return torch.einsum('bi,bi->b', emb1, emb2) / self.tau
            elif emb1.dim() == 3:
                return torch.einsum('bji,bji->bj', emb1, emb2) / self.tau
        elif self.similarity_metric == 'l2':
            return -torch.norm(emb1 - emb2, dim=-1) / self.tau
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        
class TripletLoss(nn.Module):
    def __init__(self, tau=0.1, similarity='cosine'):
        """
        InfoNCE loss for contrastive learning.
        Args:
            temperature (float): Temperature scaling factor.
        """
        super().__init__()
        self.tau = tau
        self.similarity = similarity
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, anchor_emb, positive_emb, negative_emb):
        """
        Compute the InfoNCE loss.
        Args:
            anchor_emb (torch.Tensor): Embeddings of the anchor samples.
            positive_emb (torch.Tensor): Embeddings of the positive samples.
            negative_emb (torch.Tensor): Embeddings of the negative samples.
        Returns:
            torch.Tensor: Computed InfoNCE loss.
        """
        # Normalize embeddings
        anchor_emb = F.normalize(anchor_emb, dim=1)
        positive_emb = F.normalize(positive_emb, dim=1)
        negative_emb = F.normalize(negative_emb, dim=1)

        # Compute similarity scores
        if self.similarity == 'cosine':
            positive_scores = torch.sum(anchor_emb * positive_emb, axis=-1) / self.tau
            negative_scores = torch.sum(anchor_emb * negative_emb, axis=-1) / self.tau
        elif self.similarity == 'l2':
            positive_scores = -torch.norm(anchor_emb - positive_emb, dim=-1) / self.tau
            negative_scores = -torch.norm(anchor_emb - negative_emb, dim=-1) / self.tau
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

        # Combine scores and create labels
        logits = torch.stack([positive_scores, negative_scores], dim=1)
        labels = torch.zeros_like(positive_scores, device=logits.device, dtype=torch.long)

        # Compute loss
        loss = self.criterion(logits, labels)
        return loss

class InfoNCELoss(nn.Module):
    def __init__(self, tau=1):
        """
        In-batch InfoNCE loss for contrastive learning.
        Args:
            tau (float): Temperature scaling factor.
        """
        super().__init__()
        self.tau = tau

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor):
        """
        Compute the in-batch InfoNCE loss.
        Args:
            emb1 (torch.Tensor): Embeddings of the anchor samples (batch_size, embedding_dim).
            emb2 (torch.Tensor): Embeddings of the anchor samples (batch_size, embedding_dim).
        Returns:
            torch.Tensor: Computed in-batch InfoNCE loss.
        """
        assert emb1.shape == emb2.shape, "Embeddings must have the same shape"

        # Normalize embeddings
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        # Compute logits and create labels
        logits = torch.matmul(emb1, emb2.T) / self.tau
        labels = torch.eye(logits.size(0), device=logits.device)

        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return loss

class JointInfoNCELoss(nn.Module):
    def __init__(self, tau=1, similarity='cosine'):
        """
        Joint InfoNCE loss for contrastive learning.
        Args:
            tau (float): Temperature scaling factor.
        """
        super().__init__()
        self.tau = tau
        self.similarity = similarity

    def forward(self, anchor_emb: torch.Tensor,  positive_emb: torch.Tensor, negative_emb: torch.Tensor, gates: torch.Tensor):
        """
        Compute the in-batch InfoNCE loss.
        Args:
            anchor_emb (torch.Tensor): Embeddings of the anchor samples (batch_size, num_experts, embedding_dim).
            positive_emb (torch.Tensor): Embeddings of the positive samples (batch_size, num_experts, embedding_dim).
            negative_emb (torch.Tensor): Embeddings of the negative samples (batch_size, num_experts, embedding_dim).
            gates (torch.Tensor): A probability tensor of shape (batch_size, num_experts).
        Returns:
            torch.Tensor: Computed in-batch InfoNCE loss.
        """
        assert anchor_emb.shape == positive_emb.shape, "Embeddings must have the same shape"
        assert anchor_emb.shape == negative_emb.shape, "Embeddings must have the same shape"

        # Normalize embeddings
        anchor_emb = F.normalize(anchor_emb, dim=-1) # (B, E, D)
        positive_emb = F.normalize(positive_emb, dim=-1) # (B, E, D)
        negative_emb = F.normalize(negative_emb, dim=-1) # (B, E, D)
        B, E, _ = anchor_emb.shape

        logits = torch.zeros(B, 2*B, E).to(anchor_emb.device, dtype=anchor_emb.dtype) # (B, 2B, E)

        # Compute similarity scores
        if self.similarity == 'cosine':
            logits[:, :B, :] += torch.einsum('bij,pij->bpi', anchor_emb, positive_emb) / self.tau  # (B, B, E)
            logits[:, B:, :] += torch.einsum('bij,pij->bpi', anchor_emb, negative_emb) / self.tau # (B, B, E)
        elif self.similarity == 'l2':
            anchor_emb = anchor_emb.unsqueeze(0) # (1, B, E, D)
            positive_emb = positive_emb.unsqueeze(1) # (B, 1, E, D)
            negative_emb = negative_emb.unsqueeze(1) # (B, 1, E, D)
            logits[:, :B, :] += -torch.norm(anchor_emb - positive_emb, dim=-1) / self.tau  # (B, B, E)
            logits[:, B:, :] += -torch.norm(anchor_emb - negative_emb, dim=-1) / self.tau  # (B, B, E)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

        # Multiply gates with logits
        gates_tensor = gates.unsqueeze(1) # (B, 1, E)
        logits = torch.sum(logits * gates_tensor, dim=-1)  # (B, 2B)
        
        logprobs = F.log_softmax(logits, dim=-1) # (B, 2B)
        target_logprobs = logprobs[torch.arange(B), torch.arange(B)]

        # Difference and sum over i
        return - target_logprobs.mean()

class LabeledInfoNCELoss(nn.Module):
    def __init__(self, tau=1, similarity='cosine'):
        """
        Joint InfoNCE loss for contrastive learning.
        Args:
            tau (float): Temperature scaling factor.
        """
        super().__init__()
        self.tau = tau
        self.similarity = similarity

    def forward(self, anchor_emb: torch.Tensor,  other_emb: torch.Tensor, gates: torch.Tensor, labels: torch.Tensor):
        """
        Compute the in-batch InfoNCE loss.
        Args:
            anchor_emb (torch.Tensor): Embeddings of the anchor samples (batch_size, num_experts, embedding_dim).
            other_emb (torch.Tensor): Embeddings of the other samples (batch_size, num_experts, embedding_dim).
            gates (torch.Tensor): A probability tensor of shape (batch_size, num_experts).
            labels (torch.Tensor): Labels for the samples (batch_size,).
        Returns:
            torch.Tensor: Computed in-batch InfoNCE loss.
        """
        assert anchor_emb.shape == other_emb.shape, "Embeddings must have the same shape"

        # Normalize embeddings
        anchor_emb = F.normalize(anchor_emb, dim=-1) # (B, E, D)
        other_emb = F.normalize(other_emb, dim=-1) # (B, E, D)
        B, E, _ = anchor_emb.shape

        logits = torch.zeros(B, B, E).to(anchor_emb.device, dtype=anchor_emb.dtype) # (B, B, E)

        # Compute similarity scores
        if self.similarity == 'cosine':
            logits += torch.einsum('bij,pij->bpi', anchor_emb, other_emb) / self.tau  # (B, B, E)
        elif self.similarity == 'l2':
            anchor_emb = anchor_emb.unsqueeze(0) # (1, B, E, D)
            other_emb = other_emb.unsqueeze(1) # (B, 1, E, D)
            logits += -torch.norm(anchor_emb - other_emb, dim=-1) / self.tau  # (B, B, E)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")

        # Multiply gates with logits
        gates_tensor = gates.unsqueeze(1) # (B, 1, E)
        logits = torch.sum(logits * gates_tensor, dim=-1)  # (B, B)
        
        logprobs = F.log_softmax(logits, dim=-1) # (B, B)
        target_logprobs = logprobs[torch.arange(B), torch.arange(B)] # (B,)

        # Mean over the positive samples
        return - target_logprobs[labels == 1].mean()
    

class InBatchInfoNCELoss(nn.Module):
    def __init__(self, tau=1, similarity='cosine', additive_margin=0):
        """
        Joint InfoNCE loss for contrastive learning.
        Args:
            tau (float): Temperature scaling factor.
        """
        super().__init__()
        self.tau = tau
        self.similarity = similarity
        self.additive_margin = additive_margin

    def forward(self, anchor_emb: torch.Tensor,  target_emb: torch.Tensor):
        """
        Compute the in-batch InfoNCE loss.
        Args:
            anchor_emb (torch.Tensor): Embeddings of the anchor samples (batch_size, embedding_dim).
            target_emb (torch.Tensor): Embeddings of the other samples (batch_size, embedding_dim)
        Returns:
            torch.Tensor: Computed in-batch InfoNCE loss.
        """
        assert anchor_emb.shape == target_emb.shape, "Embeddings must have the same shape"

        # Normalize embeddings
        anchor_emb = F.normalize(anchor_emb, dim=-1) # (B, D)
        target_emb = F.normalize(target_emb, dim=-1) # (B, D)
        B, _ = anchor_emb.shape

        logits = torch.zeros(B, B).to(anchor_emb.device, dtype=anchor_emb.dtype) # (B, B)

        # Compute similarity scores
        if self.similarity == 'cosine':
            logits += torch.einsum('bj,pj->bp', anchor_emb, target_emb) / self.tau  # (B, B)
        elif self.similarity == 'l2':
            anchor_emb = anchor_emb.unsqueeze(0) # (1, B, D)
            target_emb = target_emb.unsqueeze(1) # (B, 1, D)
            logits += -torch.norm(anchor_emb - target_emb, dim=-1) / self.tau  # (B, B)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity}")
        
        logits[torch.arange(B), torch.arange(B)] -= self.additive_margin
        
        logprobs = F.log_softmax(logits, dim=-1) # (B, B)
        target_logprobs = logprobs[torch.arange(B), torch.arange(B)] # (B,)

        # Mean over the positive samples
        return - target_logprobs.mean()