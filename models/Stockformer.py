import os
import torch
import torch.nn as nn

from layers.encoder import Encoder, EncoderLayer, ConvLayer
from layers.attn import FullAttention, AttentionLayer, ProbAttention
from layers.embed import DataEmbedding
from utils.masking import QuestionMask


class Stockformer(nn.Module):
    def __init__(self, config):
        super(Stockformer, self).__init__()
        self.pred_len = config.pred_len
        assert self.pred_len == 1, "Stockformer needs pred_len to be 1"
        self.attn = config.attn
        self.output_attention = config.output_attention

        self.seq_len = config.seq_len

        self.final_mode = config.final_mode

        # Embedding
        self.enc_embedding = DataEmbedding(
            config.enc_in,
            config.d_model,
            config.t_embed,
            config.freq,
            config.dropout_emb,
            emb_t2v_app_dim=config.emb_t2v_app_dim,
        )
        # Attention
        Attn = ProbAttention if config.attn == "prob" else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            True if config.final_mode == "mode3" else False,
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
                    ln_mode=config.ln_mode,
                )
                for l in range(config.e_layers)
            ],
            [ConvLayer(config.d_model) for l in range(config.e_layers - 1)]
            if config.distil
            else None,
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        if config.final_mode == "mode1":
            self.final = nn.Linear(
                config.d_model * config.seq_len, config.c_out, bias=True
            )
        elif config.final_mode == "mode2" or config.final_mode == "mode3":
            self.final = nn.Linear(config.d_model, config.c_out, bias=True)
        else:
            raise Exception(f"Invalid final_mode: {config.final_mode}")
        # nn.init.xavier_normal_(self.final.weight, gain=nn.init.calculate_gain("tanh"))

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

        if self.final_mode == "mode3":
            # This gives the encoder a question input as the last token
            # TODO: Maybe this should be initialized differently, like to the mean of x_enc, random, mean of dataset?
            zeros = torch.zeros([x_enc.shape[0], 1, x_enc.shape[2]]).to(x_enc)
            x_enc = torch.cat([x_enc, zeros], 1)
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec], 1)
            assert enc_self_mask is None
            enc_self_mask = QuestionMask(
                x_enc.shape[0], x_enc.shape[1], device=x_enc.device
            )

        # emb_out is (batch_size / num gpus, seq_len, d_model)
        emb_out = self.enc_embedding(x_enc, x_mark_enc)

        # enc_out is (batch_size / num gpus, seq_len, d_model) but seq_len will change if distil
        enc_out, attns = self.encoder(emb_out, attn_mask=enc_self_mask)

        if self.final_mode == "mode1":
            out = self.final(enc_out.flatten(start_dim=1))
        elif self.final_mode == "mode2" or self.final_mode == "mode3":
            out = self.final(enc_out[:, -1, :])
        else:
            assert False, f"Forward missing valid final mode {self.final_mode}"

        # The None below is just adding a dummy dimension
        if self.output_attention:
            return out[:, None, :], attns
        else:
            return out[:, None, :]  # (batch_size, 1, c_out)
