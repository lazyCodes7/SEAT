import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import normal
from .attention import TanhAttention
class AttentionLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_size,
        num_classes,
        dropout = 0.4,
        lstm_layer = 2

    ):
        super(AttentionLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(input_size = emb_dim, hidden_size = hidden_size, bidirectional = True)
        self.attention = TanhAttention(hidden_size = hidden_size*2)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size*lstm_layer, hidden_size*lstm_layer),
                                 nn.BatchNorm1d(hidden_size*lstm_layer),
                                 nn.ReLU())
        self.fc2 = nn.Linear(hidden_size*lstm_layer, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.m = normal.Normal(0, 1e-3)

    def forward(self, x, x_len):
        x = self.embedding(x)
        x = self.dropout(x)
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        out1, (h_n, c_n) = self.lstm(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        x, attn = self.attention(x, lengths) # skip connect


        y = self.fc1(self.dropout(x))
        y = self.fc2(self.dropout(y))
        y = self.sigmoid(y.squeeze())
        return y

    def atten_forward(self, x, x_len):
        x = self.embedding(x)
        x = self.dropout(x)
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        out1, (h_n, c_n) = self.lstm(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        x, _ = self.attention(x, lengths) # skip connect
        y = self.fc1(self.dropout(x))
        y = self.fc2(self.dropout(y))
        y = self.sigmoid(y.squeeze())
        return x, y

    def perturb(self, x, x_len, device = 'cpu'):
      x = self.embedding(x)
      idx = torch.randint(low = 0, high = x.shape[1], size = (x.shape[0],))
      s = self.m.sample((x.shape[0], 1, x.shape[2]))
      x[:,idx,:] = x[:,idx,:] + s.to(device)
      x = self.dropout(x)
      x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
      out1, (h_n, c_n) = self.lstm(x)
      x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
      x, attn = self.attention(x, lengths) # skip connect
      y = self.fc1(self.dropout(x))
      y = self.fc2(self.dropout(y))
      y = self.sigmoid(y.squeeze())
      return x, y
