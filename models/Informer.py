import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from layers.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from layers.decoder import Decoder, DecoderLayer
from layers.attn import FullAttention, ProbAttention, AttentionLayer
from layers.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, config):
        # enc_in, dec_in, c_out, seq_len, label_len, out_len,
        #         factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
        #         dropout=0.0, attn='prob', t_embed='fixed', freq='h', activation='gelu',
        #         output_attention = False, distil=True, mix=True,
        #         device=torch.device('cuda:0')
        super(Informer, self).__init__()
        self.pred_len = config.pred_len
        self.attn = config.attn
        self.output_attention = config.output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(
            config.enc_in, config.d_model, config.t_embed, config.freq, config.dropout_emb
        )
        self.dec_embedding = DataEmbedding(
            config.dec_in, config.d_model, config.t_embed, config.freq, config.dropout_emb
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
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=False,
                        ),
                        config.d_model,
                        config.n_heads,
                        mix=config.mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=False,
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
                for l in range(config.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )
        self.projection = nn.Linear(config.d_model, config.c_out, bias=True)

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
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, config):
        # enc_in, dec_in, c_out, seq_len, label_len, out_len,
        #         factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512,
        #         dropout=0.0, attn='prob', t_embed='fixed', freq='h', activation='gelu',
        #         output_attention = False, distil=True, mix=True,
        #         device=torch.device('cuda:0'))
        super(InformerStack, self).__init__()
        self.pred_len = config.pred_len
        self.attn = config.attn
        self.output_attention = config.output_attention

        assert (
            type(config.e_layers) is list
        ), "For Informer Stack e_layers must be a list"

        # Encoding
        self.enc_embedding = DataEmbedding(
            config.enc_in, config.d_model, config.t_embed, config.freq, config.dropout_emb
        )
        self.dec_embedding = DataEmbedding(
            config.dec_in, config.d_model, config.t_embed, config.freq, config.dropout_emb
        )
        # Attention
        Attn = ProbAttention if config.attn == "prob" else FullAttention
        # Encoder

        inp_lens = list(
            range(len(config.e_layers))
        )  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
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
                    for l in range(el)
                ],
                [ConvLayer(config.d_model) for l in range(el - 1)]
                if config.distil
                else None,
                norm_layer=torch.nn.LayerNorm(config.d_model),
            )
            for el in config.e_layers
        ]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            True,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=False,
                        ),
                        config.d_model,
                        config.n_heads,
                        mix=config.mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=False,
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
                for l in range(config.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        self.projection = nn.Linear(config.d_model, config.c_out, bias=True)

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
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )
        dec_out = self.projection(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]
