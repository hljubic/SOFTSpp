
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_invertedOld(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.AlphaDropout(p=dropout)  # nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)

        if self.training and torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * 0.1
            x = x + noise  # Add Gaussian noise to the input

        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        # x: [Batch Variate d_model]
        return self.dropout(x)  # probati x = self.dropout(nn.GELU()(x))


class PositionalEmbedding(nn.Module):
    def __init__(self, d_series, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_series), requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_series, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_series))
        self.position_embedding[:, :, 0::2] = torch.sin(position * div_term)
        self.position_embedding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.position_embedding[:, :x.size(1)]
        return x


import math


class LogarithmicPositionalEmbedding(nn.Module):
    def __init__(self, d_series, max_len=5000):
        super(LogarithmicPositionalEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_series), requires_grad=False)
        position = torch.arange(1, max_len + 1).unsqueeze(1).float()  # Avoid log(0) by starting from 1
        div_term = torch.exp(torch.arange(0, d_series, 2).float() * -(math.log(10000.0) / d_series))

        # Compute log-scaled positional embeddings
        log_position = torch.log(position)
        self.position_embedding[:, :, 0::2] = torch.sin(log_position * div_term)
        self.position_embedding[:, :, 1::2] = torch.cos(log_position * div_term)

    def forward(self, x):
        x = x + self.position_embedding[:, :x.size(1)]
        return x
