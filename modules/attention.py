import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import normal
import lightning.pytorch as pl

def masked_softmax(attn_odds, masks) :
    attentions = torch.softmax(F.relu(attn_odds.squeeze()), dim=-1)

    # apply mask and renormalize attention scores (weights)
    masked = attn_odds * masks
    _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

    attn_odds = masked.div(_sums)
    return attn_odds
def perturbed_masked_softmax(attn_odds, masks, delta) :
    attentions = torch.softmax(F.relu(attn_odds.squeeze()), dim=-1)
    attentions = attentions + delta
    attentions = torch.clamp(attentions, min=0, max=1)

    # apply mask and renormalize attention scores (weights)
    masked = attn_odds * masks

    _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

    attn_odds = masked.div(_sums)
    return attn_odds

class TanhAttention(pl.LightningModule):
    def __init__(self, hidden_size):
        super(TanhAttention, self).__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)
        self.num_hiddens = hidden_size//2
        self.compute = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self, hidden, lengths):
        #input_seq = (B, L), hidden : (B, L, H), masks : (B, L)
        max_len = hidden.shape[1]
        attn1 = nn.Tanh()(self.attn1(hidden))
        attn2 = self.attn2(attn1).squeeze(-1)
        masks = torch.ones(attn2.size(), requires_grad=False).to(self.compute)
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                masks[i, l:] = 0
        attn = masked_softmax(attn2, masks)
        # apply attention weights
        weighted = torch.mul(hidden, attn.unsqueeze(-1).expand_as(hidden))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attn
