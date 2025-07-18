# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from mixture_of_adapters.layers import LowRankLinearLayer

class Adapter(nn.Module):
    """A linear adapter module that transforms embeddings using a low-rank linear layer.
    
    Args:
        input_size: integer - size of the input embeddings
        output_size: integer - size of the output embeddings
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        hidden_size = int(np.sqrt(input_size * output_size))

        # instantiate layer
        self.adapter = nn.Sequential(
            nn.RMSNorm(self.input_size, elementwise_affine=False),
            nn.Dropout(0.3),
            nn.Linear(self.input_size, hidden_size),
            nn.RMSNorm(hidden_size, elementwise_affine=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, self.output_size),
        )
        

    def forward(self, embeddings):
        """Transform input embeddings using the adapter.
        
        Args:
            embeddings: tensor shape [batch_size, input_size]

        Returns:
            dict: A dictionary containing the transformed embeddings with key 'embeddings'
        """
        y = self.adapter(embeddings)
        return y
    
    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Transform input embeddings using the adapter.
        
        Args:
            embeddings: tensor shape [batch_size, input_size]
            contexts: tensor shape [batch_size, context_size] - not used in this implementation
        
        Returns:
            tensor: A tensor with shape [batch_size, output_size]
        """
        device = next(self.parameters()).device
        embeddings = embeddings.to(device)
        y = self.adapter(embeddings)
        return y
    
class ConcatAdapter(nn.Module):
    """A linear adapter module that transforms embeddings using a low-rank linear layer.
    
    Args:
        input_size: integer - size of the input embeddings
        output_size: integer - size of the output embeddings
    """

    def __init__(self, input_size, output_size, context_size=None):
        super().__init__()
        self.input_size = input_size
        if context_size is None:
            context_size = input_size
        self.context_size = context_size
        self.output_size = output_size

        self.transform = nn.Sequential(
            nn.RMSNorm(self.context_size, elementwise_affine=False),
            nn.Linear(self.context_size, self.input_size),
        )
        
        self.adapter = nn.Sequential(
            nn.Linear(self.input_size + self.context_size, self.output_size),
        )
        
    def encode(self, embeddings: torch.Tensor, contexts: torch.Tensor):
        embeddings = self.transform(embeddings)
        contexts = self.transform(contexts)
        y = self.adapter(torch.cat([embeddings, contexts], dim=1))
        return y
    
    def forward(self, embeddings: torch.Tensor, contexts: torch.Tensor):
        return self.encode(embeddings, contexts)
    
class HadamardAdapter(nn.Module):
    """A linear adapter module that transforms embeddings using a low-rank linear layer.
    
    Args:
        input_size: integer - size of the input embeddings
        output_size: integer - size of the output embeddings
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.transform = nn.Sequential(
            nn.RMSNorm(self.input_size, elementwise_affine=False),
            nn.Linear(self.input_size, self.input_size),
        )

        self.adapter = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
        )
        
    def encode(self, embeddings: torch.Tensor, contexts: torch.Tensor):
        embeddings = self.transform(embeddings)
        contexts = self.transform(contexts)
        y = self.adapter(embeddings * contexts)
        return y
    
    def forward(self, embeddings: torch.Tensor, contexts: torch.Tensor):
        return self.encode(embeddings, contexts)
    
class HyperCLAdapter(nn.Module):
    """A linear adapter module that transforms embeddings using a low-rank linear layer.
    
    Args:
        input_size: integer - size of the input embeddings
        output_size: integer - size of the output embeddings
    """

    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        if hidden_size is None:
            self.hypernetwork = nn.Sequential(
                nn.Linear(self.input_size, self.input_size * self.output_size),
            )
        else:
            self.hypernetwork = nn.Sequential(
                nn.Linear(self.input_size, hidden_size),
                nn.Linear(hidden_size, self.input_size * self.output_size),
            )
        
    def encode(self, embeddings: torch.Tensor, contexts: torch.Tensor):
        projection_matrices = self.hypernetwork(contexts).reshape(-1, self.input_size, self.output_size)
        y = torch.einsum('ijk,ij->ik', projection_matrices, embeddings)
        return y
    
    def forward(self, embeddings: torch.Tensor, contexts: torch.Tensor):
        return self.encode(embeddings, contexts)
