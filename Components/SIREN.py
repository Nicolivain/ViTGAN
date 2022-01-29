import numpy as np
import torch
import torch.nn as nn


class SIREN(nn.Module):
    def __init__(self, in_features, out_features, bias = True, is_first = False, omega_0 = 30, **kwargs):
        """
        Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN
        :param in_features: number of input features
        :param out_features: number of output features
        :param bias: add a bias or not to the linear transformation
        :param is_first: first layer
        :param omega_0: pulsation of the sine activation
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
