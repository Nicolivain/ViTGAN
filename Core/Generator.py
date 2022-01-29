import torch
import torch.nn as nn

from Components.MLP import MLP
from Components.Tranformer import TransformerSLN
from Components.SLN import SLN
from Components.SIREN import SIREN


class Generator(nn.Module):
    def __init__(self, lattent_size, img_size, n_channels, feature_hidden_size,  n_transformer_layers=1, mapping_mlp_params=None, transformer_params=None, output_net_params=None):
        super(Generator, self).__init__()

        self.lattent_size          = lattent_size
        self.img_size              = img_size
        self.feature_hidden_size   = feature_hidden_size
        self.n_channels             = n_channels
        self.n_transformer_layers  = n_transformer_layers

        self.mapping_params        = {} if mapping_mlp_params is None else mapping_mlp_params
        self.transformer_params    = {} if transformer_params is None else transformer_params
        self.output_net_params     = {} if output_net_params is None else output_net_params

        self.mapping_params['in_features'], self.mapping_params['out_features'] = self.lattent_size, self.img_size*self.feature_hidden_size
        self.mapping_mlp = MLP(**self.mapping_params)

        self.emb = torch.nn.Parameter(torch.randn(self.img_size, self.feature_hidden_size))

        self.transformer_params['in_features'] = self.feature_hidden_size
        self.transformer_layers = nn.ModuleList([TransformerSLN(**self.transformer_params) for _ in range(self.n_transformer_layers)])

        self.sln = SLN(self.feature_hidden_size)

        self.output_net = nn.Sequential(
            SIREN(self.feature_hidden_size, 2*img_size, is_first=True),
            SIREN(2*self.img_size, self.img_size*self.n_channels)
        )

    def forward(self, x):
        w = self.mapping_mlp(x).view(-1, self.img_size, self.feature_hidden_size)
        h = self.emb
        for tf in self.transformer_layers:
            h = tf(h, w)
        w = self.sln(h, w)
        res = self.output_net(w).view(x.shape[0], self.n_channels, self.img_size, self.img_size)
        return res


if __name__ == '__main__':
    ipt = torch.randn(10, 1024)
    mod = Generator(1024, 64, 3, 384)
    ret = mod.forward(ipt)
    print(ret.shape)
