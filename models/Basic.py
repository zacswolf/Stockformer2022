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
        assert config.seq_len in [None, 1]
        assert config.pred_len in [None, 1]

        self.e_layers = config.e_layers
        self.enc_in = config.enc_in
        self.d_model = config.d_model
        self.c_out = config.c_out

        if self.e_layers == 1:
            layers = [nn.Linear(self.enc_in, self.c_out)]

        else:
            assert self.e_layers >= 2

            layers = [nn.Linear(self.enc_in, self.d_model), nn.GELU()]
            for _ in range(config.e_layers - 2):
                layers.append(nn.Linear(self.d_model, self.d_model))
                layers.append(nn.GELU())

            layers.append(nn.Linear(self.d_model, self.c_out))

        self.model = nn.Sequential(*layers)

    def forward(self, x, *args):
        # x: [Batch, Input length, Channel]
        assert x.shape[1] == 1
        return self.model(x)


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, config):
        super(NLinear, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x, *args):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]
