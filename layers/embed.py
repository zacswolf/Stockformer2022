import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TokenEmbeddingBasic(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbeddingBasic, self).__init__()
        self.linear = nn.Linear(c_in, d_model)

    def forward(self, x):
        x = self.linear(x)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, t_embed="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if t_embed == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, t_embed="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class Time2Vec(nn.Module):
    def __init__(self, time_emb_dim, freq="h"):
        super(Time2Vec, self).__init__()
        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        time_feat_dim = freq_map[freq]

        self.output_dim = time_emb_dim

        self.out_features = time_emb_dim

        # TODO: Initialize uniform
        self.linear_periodic = nn.Linear(time_feat_dim, time_emb_dim - 1)
        self.linear_non_periodic = nn.Linear(time_feat_dim, 1)

    def forward(self, x):
        non_periodic = self.linear_non_periodic(x.float())
        periodic = torch.sin(self.linear_periodic(x.float()))
        out = torch.cat([non_periodic, periodic], -1)
        return out


class DataEmbedding(nn.Module):
    def __init__(
        self,
        c_in,
        d_model,
        t_embed="fixed",
        freq="h",
        dropout_emb=0.01,
        position_embedding=True,
        emb_t2v_app_dim=32,
        tok_emb="default",
    ):
        super(DataEmbedding, self).__init__()

        self.append_time_emb = t_embed == "time2vec_app"

        # For the temporal embedding
        if t_embed is not None:
            assert t_embed in [
                "fixed",
                "learned",
                "timeF",
                "time2vec_add",
                "time2vec_app",
            ], "Invalid t_embed"
            if t_embed == "fixed" or t_embed == "learned":
                self.temporal_embedding = TemporalEmbedding(
                    d_model=d_model, t_embed=t_embed, freq=freq
                )
            elif t_embed == "timeF":
                self.temporal_embedding = TimeFeatureEmbedding(
                    d_model=d_model, t_embed=t_embed, freq=freq
                )
            elif t_embed == "time2vec_add":
                # Time2Vec time embedding add elementwise
                self.temporal_embedding = Time2Vec(time_emb_dim=d_model, freq=freq)
            elif t_embed == "time2vec_app":
                # Time2Vec time embedding appended
                assert (
                    emb_t2v_app_dim is not None
                ), "Need to provide the emb_t2v_app_dim argument"
                assert emb_t2v_app_dim > 0 and emb_t2v_app_dim < d_model
                self.temporal_embedding = Time2Vec(
                    time_emb_dim=emb_t2v_app_dim, freq=freq
                )
                d_model -= emb_t2v_app_dim
        else:
            self.temporal_embedding = lambda _: 0

        # For the value embedding
        if tok_emb == "basic":
            self.value_embedding = TokenEmbeddingBasic(c_in=c_in, d_model=d_model)
        elif tok_emb == "raw":
            self.value_embedding = lambda x: x
            assert c_in == d_model, "c_in and d_model must be equal for raw embedding"
            assert (
                t_embed != "time2vec_app"
            ), "time2vec_app not supported for raw embedding"
        else:
            self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        self.position_embedding = (
            PositionalEmbedding(d_model=d_model) if position_embedding else lambda x: 0
        )

        self.dropout = nn.Dropout(p=dropout_emb)

    def forward(self, x, x_mark):
        if self.append_time_emb:
            x = self.value_embedding(x) + self.position_embedding(x)
            x_drop = self.dropout(x)
            time_emb = self.temporal_embedding(x_mark)
            return torch.concat([x_drop, time_emb], -1)
        else:
            x = (
                self.value_embedding(x)
                + self.position_embedding(x)
                + self.temporal_embedding(x_mark)
            )
            return self.dropout(x)
