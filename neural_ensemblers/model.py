import copy

import numpy as np
import torch 
import torch.nn as nn

class NeuralEnsembler(nn.Module):  # Sample as Sequence
    def __init__(
        self,
        num_base_functions=1,
        hidden_dim=32,
        output_dim=1,
        num_layers=3,
        dropout_rate=0,
        num_heads=1,
        inner_batch_size=10,
        omit_output_mask=False,
        task_type="classification",
        mode="model_averaging",
        num_classes=None,
        **kwargs
    ):
        super().__init__()

        # input = [BATCH SIZE X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS X NUMBER OF CLASSES]
        self.num_base_functions = num_base_functions
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.inner_batch_size = inner_batch_size
        self.omit_output_mask = omit_output_mask
        self.mode = mode
        self.task_type = task_type
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        if self.mode == "model_averaging":
            num_layers=-1

        if self.mode == "stacking":
            self.output_dim = 1
        else:
            self.output_dim = num_base_functions

        self.build_modules()

    def build_modules(self):
        """
        Build the modules for the neural network.
        """
        layers_dim = [self.num_base_functions] + [self.hidden_dim] * (self.num_layers-1)
        first_module = []
        for i in range(len(layers_dim) - 1):
            first_module.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            first_module.append(nn.ReLU())
        self.first_module = nn.Sequential(*first_module)
        self.out_layer = nn.Linear(self.hidden_dim, self.output_dim)

        if self.mode == "model_averaging":
            self.second_module = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU())


    def get_batched_weights(self, x):

        num_samples, num_classes, num_base_functions = x.shape

        # [NUMBER OF CLASSES X NUMBER OF SAMPLES X NUMBER OF BASE FUNCTIONS]
        x = self.first_module(x)

        return x


    def get_mask_and_scaling_factor(self, num_base_functions, device):
        mask= (torch.rand(size=(num_base_functions,)) > self.dropout_rate).float().to(device)
        scaling_factor = 1./(1.- self.dropout_rate)
        return mask, scaling_factor

    def forward(
        self, x
    ):
        # X = [NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        num_samples, num_classes, num_base_functions = x.shape
        base_functions = copy.deepcopy(x)

        if self.training and self.dropout_rate > 0:
            mask, scaling_factor = self.get_mask_and_scaling_factor(num_base_functions, x.device)
            for i, dim in enumerate([ num_samples, num_classes]):
                mask = torch.repeat_interleave(
                    mask.unsqueeze(i), dim, dim=i
                )
        else:
            mask = None

        w = []
        idx = np.arange(num_classes)
        for i in range(0, num_classes, self.inner_batch_size):
            range_idx = idx[range(i, min(i + self.inner_batch_size, num_classes))]
            if mask is not None:
                temp_x = (x[:,range_idx]*mask[:,range_idx])*scaling_factor
                base_functions[:,range_idx] = base_functions[:,range_idx]*mask[:,range_idx]
            else:
                temp_x = x[:,range_idx]

            temp_w = self.get_batched_weights(x = temp_x)
            w.append(temp_w)

        w = torch.cat(w, axis=1)

        if self.mode == "model_averaging":
            w = w.mean(axis=1)
            w = self.second_module(w)
            w = torch.repeat_interleave(
                w.unsqueeze(1), num_classes, dim=1
            )

        w = self.out_layer(w)

        if self.mode == "stacking":
            x = w.squeeze(-1)
            if self.task_type == "classification":
                x = torch.nn.functional.softmax(x, dim=-1)
            return x, None

        elif self.mode == "model_averaging":
            if (mask is not None) and (not self.omit_output_mask):
                w = w.masked_fill(mask == 0, -1e9)

            # num_samples, num_classes, num_base_functions = w.shape
            w_norm = torch.nn.functional.softmax(w, dim=-1)
            x = torch.multiply(base_functions, w_norm).sum(axis=-1)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
            # x.shape: [BATCH_SIZE, NUM_SAMPLES, NUM_CLASSES]
            # w_norm.shape : [BATCH SIZE X NUMBER OF SAMPLES  X NUMBER OF CLASSES X NUMBER OF BASE FUNCTIONS]
        return x, w_norm