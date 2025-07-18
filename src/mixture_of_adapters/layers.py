# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class LowRankLinearLayer(nn.Module):
    """
    A low-rank linear layer module.
    This module performs a linear transformation using a low-rank approximation.

    Args:
        input_size: int - size of the input features.
        output_size: int - size of the output features.
        rank: int - rank of the low-rank approximation.
    """
    def __init__(self, input_size, output_size, rank):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank

        # Low-rank decomposition using linear layers
        self.u = nn.Linear(input_size, rank, bias=False)
        self.v = nn.Linear(rank, output_size, bias=False)
        nn.init.orthogonal_(self.u.weight)
        nn.init.orthogonal_(self.v.weight)

    def forward(self, x):
        return self.v(self.u(x))
