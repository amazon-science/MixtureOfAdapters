# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import torch
import os
from collections import defaultdict
import lightning.pytorch as pl

class MetricsPlotCallback(pl.Callback):
    def __init__(self, save_dir="plots"):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_history = {}

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Ensure all keys in metrics_history are initialized as lists
        for key in metrics.keys():
            if key not in self.metrics_history:
                self.metrics_history[key] = []

        # Append the current epoch's metrics as tuples (epoch, value)
        for key, value in metrics.items():
            value = value.cpu().item() if isinstance(value, torch.Tensor) else value
            self.metrics_history[key].append((epoch, value))

        self.plot_metrics(epoch)

    def plot_metrics(self, epoch):

        # Group metrics by their names
        grouped_metrics = defaultdict(list)
        for key in self.metrics_history:
            metric_name = key.split('/')[0] # Extract the metric name (e.g., "triplet_loss")
            grouped_metrics[metric_name].append(key)

        # Number of subplots: one for each group of metrics
        ncols = len(grouped_metrics)
        nrows = 2 if ncols > 3 else 1
        ncols = (ncols + 1) // 2 if nrows == 2 else ncols
        fig, ax = plt.subplots(figsize=(6 * ncols, 6 * nrows), nrows=nrows, ncols=ncols)
        ax = ax.flatten() if nrows > 1 else ax

        metric_names = list(grouped_metrics.keys())
        metric_names.remove("triplet_loss")
        metric_names = ["triplet_loss"] + metric_names

        # Loop through grouped metrics and create a plot for each group
        for i, metric_name in enumerate(metric_names):
            keys = grouped_metrics[metric_name]
            current_ax = ax[i] if ncols > 1 else ax
            current_ax.set_title(f"{metric_name}")
            for key in keys:
                epochs, values = zip(*self.metrics_history[key])
                current_ax.plot(epochs, values, label=key)
            current_ax.set_xlabel("Epoch")
            current_ax.set_ylabel("Value")
            current_ax.legend()
            current_ax.grid()

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"metrics.png"))
        plt.close()
