import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='relu', dropout_rate=0.0):
        super().__init__()

        self.layers = layers if layers is not None else []
        self.model = nn.ModuleList([
            nn.Sequential(nn.Linear(lp, lnext), nn.Dropout(dropout_rate))
            for lp, lnext in zip([in_features] + self.layers, self.layers + [out_features])
            ])

        self.act = torch.nn.ReLU() if activation == 'relu' else torch.nn.Tanh() if activation == 'tanh' else torch.nn.Sigmoid() if activation == 'sigmoid' else ValueError

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
            x = self.act(x)
        return x
