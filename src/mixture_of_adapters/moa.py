# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
from mixture_of_adapters.layers import LowRankLinearLayer


def topk_gumbel(logits: torch.Tensor, k: int, tau: float = 1, dim=-1):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    noisy_logits = logits + tau * gumbel_noise
    top_logits, topk_indices = torch.topk(noisy_logits, k=k, dim=dim)
    return top_logits, topk_indices


class MixtureOfAdapters(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    context_size: integer - size of the context
    num_experts: integer - number of experts
    top_k: integer - number of experts to use
    expert_hidden_size: integer - size of the hidden layer of the experts
    gate_hidden_size: integer - size of the hidden layer of the gating network
    shared_adapter_hidden_size: integer - size of the hidden layer of the shared adapter
    composition_mode: string - how to combine the outputs of the experts. Can be "sum" or "concat".
    """

    def __init__(self, 
                 input_size, 
                 output_size, 
                 context_size,
                 num_experts, 
                 top_k = None, 
                 expert_hidden_size=None, 
                 gate_hidden_size=None,
                 shared_adapter_hidden_size = None,
                 dropout=0.0,
                 input_noise_std=0.0,
                 context_noise_std=0.0,
                 gate_activation="linear",
                 composition_mode="sum",
        ):
        super().__init__()
        # Size parameters
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.gate_hidden_size = gate_hidden_size
        self.shared_adapter_hidden_size = shared_adapter_hidden_size
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size

        # Regularization parameters
        self.dropout = dropout
        self.input_noise_std = input_noise_std
        self.context_noise_std = context_noise_std

        if top_k is None:
            self.top_k = num_experts
        else:
            self.top_k = top_k

        self.composition_mode = composition_mode
        assert self.composition_mode in ["sum", "concat"], f"Invalid composition mode: {self.composition_mode}"
        
        self.gate_activation = gate_activation
        assert self.gate_activation in ["linear", "softmax"], f"Invalid gate activation: {self.gate_activation}"

        self.tau = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=False)

        # Shared adapter network
        if self.shared_adapter_hidden_size is not None:
            self.shared_adapter = nn.Sequential(
                nn.Linear(self.input_size, self.shared_adapter_hidden_size),
                nn.Dropout(self.dropout),
            )
        else:
            self.shared_adapter = None
        
        expert_input_dim = self.shared_adapter_hidden_size if self.shared_adapter_hidden_size is not None else self.input_size
        
        # Experts
        if self.composition_mode == "concat":
            # If concatenating, the output size of each expert is output_size / num_experts
            assert self.output_size % self.num_experts == 0, f"Output size {self.output_size} must be divisible by number of experts {self.num_experts} for concatenation."
            expert_output_dim = self.output_size // self.num_experts
        else:
            # If summing, the output size of each expert is output_size
            expert_output_dim = self.output_size

        if self.expert_hidden_size is not None:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(expert_input_dim, self.expert_hidden_size),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.expert_hidden_size, expert_output_dim),
                ) for _ in range(self.num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(expert_input_dim, expert_output_dim),
                    nn.Dropout(self.dropout),
                ) for _ in range(self.num_experts)
            ])
        
        # Gating mechanism
        if self.gate_hidden_size is not None:
            self.gate_network = nn.Sequential(
                nn.Linear(self.context_size, self.gate_hidden_size),
                nn.Dropout(self.dropout),
                nn.Linear(self.gate_hidden_size, self.num_experts),
            )
        else:
            self.gate_network = nn.Sequential(
                nn.Linear(self.context_size, self.num_experts),
            )
        
        assert(self.top_k <= self.num_experts)

    def set_tau(self, tau: float):
        """Set the temperature for the Gumbel softmax."""
        self.tau.data = torch.tensor(tau, dtype=torch.float32)

    def forward_gate(self, contexts: torch.Tensor):
        """Top-k gating.
          Args:
            contexts: input Tensor with shape [batch_size, input_size]
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
        """
        if self.training and self.context_noise_std > 0:
            noise = Normal(0, self.context_noise_std).sample(contexts.shape)
            contexts = contexts + noise.to(contexts.device)

        gates : torch.Tensor = self.gate_network(contexts)
        if self.gate_activation == "softmax":
            gates = F.softmax(gates / self.tau, dim=-1)
        elif self.gate_activation == "linear":
            gates = gates
        else:
            raise ValueError(f"Invalid gate activation: {self.gate_activation}")
        return gates
    
    def forward_experts(self, embeddings: torch.Tensor):
        """Compute the output of the experts.
        Args:
            embeddings: a Tensor with shape [batch_size, input_size]
        Returns:
            expert_outputs: a Tensor with shape [batch_size, num_experts, output_size]
        """

        if self.training and self.input_noise_std > 0:
            noise = Normal(0, self.input_noise_std).sample(embeddings.shape)
            embeddings = embeddings + noise.to(embeddings.device)

        if self.shared_adapter is not None:
            embeddings = self.shared_adapter(embeddings)
        
        expert_outputs = torch.stack([self.experts[i](embeddings) for i in range(self.num_experts)], dim=1)

        return expert_outputs
    
    def forward_adapter(self, embeddings: torch.Tensor, gates: torch.Tensor, return_expert_outputs=False):
        """Compute the output of the adapter.
        Args:
            embeddings: a Tensor with shape [batch_size, input_size]
            gates: a Tensor with shape [batch_size, num_experts]
            return_expert_outputs: boolean - whether to return the expert outputs
        Returns:
            y: a Tensor with shape [batch_size, output_size]
        """
        expert_outputs = self.forward_experts(embeddings)
        if self.composition_mode == "concat":
            y = gates.unsqueeze(-1) * expert_outputs
            y = y.view(y.shape[0], -1)
        elif self.composition_mode == "sum":
            y = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=1)
        else:
            raise ValueError(f"Invalid composition mode: {self.composition_mode}")
        if return_expert_outputs:
            return y, expert_outputs
        else:
            return y
    
    def encode(self, embeddings: torch.Tensor, contexts: torch.Tensor):
        gates = self.forward_gate(contexts)
        y = self.forward_adapter(embeddings, gates)
        return y

    def forward(self, embeddings: torch.Tensor, contexts: torch.Tensor):
        return self.encode(embeddings, contexts)
