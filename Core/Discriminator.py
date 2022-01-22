import torch
import torch.nn as nn

from Components.PatchEncoder import PatchEncoder
from Components.Tranformer import Transformer
from Components.MLP import MLP


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):
        pass
