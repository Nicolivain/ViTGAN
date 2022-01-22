import torch
import torch.nn as nn


class SLN(nn.Module):
    def __init__(self, n_feats):
        """
        Self modulated layer norm module implementation
        :param n_feats: number of input features
        """
        super(SLN, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)  # we use this for the layer normalization
        self.beta       = nn.Parameter(torch.randn(1, 1, 1))
        self.gamma      = nn.Parameter(torch.randn(1, 1, 1))

    def forward(self, h, w):
        return self.gamma * w * self.layer_norm(h) + self.beta * w
