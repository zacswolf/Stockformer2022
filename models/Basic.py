import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.embed import Time2Vec


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

        # Time Embedding
        self.app_time_emb = config.t_embed is not None
        if self.app_time_emb:
            if config.t_embed != "time2vec_app":
                raise Exception(
                    "The only options for t_embed with mlp are null and time2vec_app"
                )
            elif not (config.emb_t2v_app_dim > 0):
                raise Exception("Need to specify a valid emb_t2v_app_dim")
            self.enc_in += config.emb_t2v_app_dim
            self.temporal_embedding = Time2Vec(
                time_emb_dim=config.emb_t2v_app_dim, freq=config.freq
            )

        flattened_enc_in = self.seq_len * self.enc_in

        if self.e_layers == 1:
            layers = [nn.Linear(flattened_enc_in, self.c_out)]
        else:
            layers = [nn.Linear(flattened_enc_in, self.d_model), nn.GELU()]
            for _ in range(self.e_layers - 2):
                layers.append(nn.Dropout(config.dropout))
                layers.append(nn.Linear(self.d_model, self.d_model))
                layers.append(nn.GELU())

            layers.append(nn.Linear(self.d_model, self.c_out))

        self.model = nn.Sequential(*layers)

    def forward(self, x, x_mark, *args):
        # x: [Batch, Input length, Channel]
        if self.app_time_emb:
            time_emb = self.temporal_embedding(x_mark)
            x = torch.concat([x, time_emb], dim=-1)

        x_flat = x.reshape(x.shape[0], 1, -1)
        return self.model(x_flat)
