import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, out_features, layers=None, activation='gelu', dropout_rate=0.0, **kwargs):
        """
        Usual MLP module
        :param in_features: number of input features
        :param out_features: number of output features
        :param layers: list of hidden layer dimensions
        :param activation: activation function
        :param dropout_rate: dropout rate
        """
        super(MLP, self).__init__()
        self.layers = layers if layers is not None else []
        self.model = nn.ModuleList([
            nn.Sequential(nn.Linear(lp, lnext), nn.Dropout(dropout_rate))
            for lp, lnext in zip([in_features] + self.layers, self.layers + [out_features])
            ])

        self.act = torch.nn.ReLU() if activation == 'relu' else torch.nn.Tanh() if activation == 'tanh' else torch.nn.Sigmoid() if activation == 'sigmoid' else torch.nn.GELU() if activation == 'gelu' else ValueError

    def forward(self, x):
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx != len(self.model) - 1:
                x = self.act(x)
        return x
