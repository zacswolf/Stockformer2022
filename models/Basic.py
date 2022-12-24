import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """
    Just your everyday neural net
    """

    def __init__(self, config):
        super(MLP, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        assert config.pred_len == 1

        self.e_layers = config.e_layers
        assert config.e_layers >= 1
        self.enc_in = config.enc_in
        self.d_model = config.d_model
        self.c_out = config.c_out

        flattened_enc_in = config.seq_len * config.enc_in

        if self.e_layers == 1:
            layers = [nn.Linear(flattened_enc_in, self.c_out)]
        else:
            layers = [nn.Linear(flattened_enc_in, self.d_model), nn.GELU()]
            for _ in range(config.e_layers - 2):
                layers.append(nn.Linear(self.d_model, self.d_model))
                layers.append(nn.GELU())

            layers.append(nn.Linear(self.d_model, self.c_out))

        self.model = nn.Sequential(*layers)

    def forward(self, x, *args):
        # x: [Batch, Input length, Channel]
        x_flat = x.reshape(x.shape[0], 1, -1)
        return self.model(x_flat)
