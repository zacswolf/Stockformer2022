import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        assert config.pred_len == 1
        assert config.label_len == 0
        # Hidden dimensions
        self.d_model = config.d_model

        # Number of hidden layers
        self.e_layers = config.e_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
            input_size=config.enc_in,
            hidden_size=config.d_model,
            num_layers=config.e_layers,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=False,
        )

        self.fc_1 = nn.Linear(config.d_model, config.d_ff)
        self.relu = nn.ReLU()
        # Readout layer
        self.fc = nn.Linear(config.d_ff, config.c_out)

    def forward(self, x, *args, **kwargs):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.e_layers, x.size(0), self.d_model).to(x)

        # Initialize cell state
        c0 = torch.zeros(self.e_layers, x.size(0), self.d_model).to(x)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!

        # out = self.relu(self.fc_1(out[:, -1, :]))
        out = self.relu(self.fc_1(self.relu(hn[-1])))

        out = self.fc(out)
        # out.size() --> 100, 10
        return out[:, None]
