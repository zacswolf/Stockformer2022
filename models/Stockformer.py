import os
import torch
import torch.nn as nn

from layers.encoder import Encoder, EncoderLayer, ConvLayer
from layers.attn import FullAttention, AttentionLayer, ProbAttention
from layers.embed import DataEmbedding


class Stockformer(nn.Module):
    def __init__(self, config):
        super(Stockformer, self).__init__()
        self.pred_len = config.pred_len
        assert self.pred_len == 1, "Stockformer needs pred_len to be 1"
        self.attn = config.attn
        self.output_attention = config.output_attention

        self.seq_len = config.seq_len

        # Embedding
        self.enc_embedding = DataEmbedding(
            config.enc_in, config.d_model, config.embed, config.freq, config.dropout
        )
        # Attention
        Attn = ProbAttention if config.attn == "prob" else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                        mix=False,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            [ConvLayer(config.d_model) for l in range(config.e_layers - 1)]
            if config.distil
            else None,
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads, dim_feedforward=config.d_ff, dropout=config.dropout, activation=config.activation)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.e_layers)

        self.final = nn.Linear(config.d_model * config.seq_len, config.c_out, bias=True)
        # self.final = nn.Sequential(*[
        #     nn.Linear(config.d_model * config.seq_len, config.d_model * 4, bias=True),
        #     nn.GELU(),
        #     nn.Linear(config.d_model * 4, config.c_out, bias=True)
        # ])

        # Load pre-trained model
        if config.load_model_path is not None:
            path = os.path.join(config.checkpoints, config.load_model_path)
            print(f"Loading Model from {path}")
            self.load_state_dict(torch.load(path))

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        pre_train=False,
    ):
        # x_enc is (batch_size / num gpus, seq_len, enc_in)
        # x_mark_enc is (batch_size / num gpus, seq_len, date-representation (7forhours)
        assert len(x_enc.shape) == 3
        assert x_enc.shape[1] == self.seq_len

        # emb_out is (batch_size / num gpus, seq_len, d_model)
        emb_out = self.enc_embedding(x_enc, x_mark_enc)

        # enc_out is (batch_size / num gpus, seq_len, d_model) but seq_len will change if distil
        enc_out, attns = self.encoder(emb_out, attn_mask=enc_self_mask)

        out = self.final(enc_out.flatten(start_dim=1))

        if self.output_attention:
            return out[:, None, :], attns
        else:
            return out[:, None, :]  # (batch_size, 1, c_out)


class StockformerVanilla(nn.Module):
    def __init__(self, config):
        super(Stockformer, self).__init__()
        self.pred_len = config.pred_len
        assert self.pred_len == 1, "Stockformer needs pred_len to be 1"
        self.output_attention = config.output_attention

        self.seq_len = config.seq_len

        # Embedding
        self.enc_embedding = DataEmbedding(
            config.enc_in, config.d_model, config.embed, config.freq, config.dropout
        )

        # TODO: Make sure this is correct
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=config.e_layers
        )

        self.final = nn.Linear(config.d_model * config.seq_len, config.c_out, bias=True)
        # self.final = nn.Sequential(*[
        #     nn.Linear(config.d_model * config.seq_len, config.d_model * 4, bias=True),
        #     nn.GELU(),
        #     nn.Linear(config.d_model * 4, config.c_out, bias=True)
        # ])

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
    ):
        # x_enc is (batch_size / num gpus, seq_len, enc_in)
        # x_mark_enc is (batch_size / num gpus, seq_len, date-representation (7forhours))
        assert x_enc.shape[1] == self.seq_len

        # emb_out is (batch_size / num gpus, seq_len, d_model)
        emb_out = self.enc_embedding(x_enc, x_mark_enc)

        # enc_out is (batch_size / num gpus, seq_len, d_model) but seq_len will change if distil
        enc_out, attns = self.encoder(emb_out, attn_mask=enc_self_mask)

        out = self.final(enc_out.flatten(start_dim=1))

        if self.output_attention:
            return out[:, None, :], attns
        else:
            return out[:, None, :]  # (batch_size, 1, c_out)
