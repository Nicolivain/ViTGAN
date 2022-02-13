import torch
import torch.nn as nn

from Components.MLP import MLP
from Components.Tranformer import TransformerSLN
from Components.SLN import SLN
from Components.SIREN import SIREN

from Tools.utils import count_params


class Generator(nn.Module):
    def __init__(self, lattent_size, img_size, n_channels, feature_hidden_size=384, n_transformer_layers=1, output_hidden_dim=1024, mapping_mlp_params=None, transformer_params=None, **kwargs):
        """
        ViT Generator Class
        :param lattent_size: number of features in the lattent space
        :param img_size: output images size, the image will be square sized
        :param n_channels: number of channel in the output images
        :param feature_hidden_size: number of features in the transformers and output layers
        :param n_transformer_layers: number of stacked transformer blocks
        :param output_hidden_dim: size of the output hidden layer (in SIREN)
        :param mapping_mlp_params: kwargs for optional parameters of the mapping MLP, mandatory args will be filled automatically
        :param transformer_params: kwargs for optional parameters of the Transformer blocks, mandatory args will be filled automatically
        """
        super(Generator, self).__init__()

        self.lattent_size          = lattent_size
        self.img_size              = img_size
        self.feature_hidden_size   = feature_hidden_size
        self.n_channels            = n_channels
        self.n_transformer_layers  = n_transformer_layers
        self.output_hidden_dim     = output_hidden_dim

        self.mapping_params        = {} if mapping_mlp_params is None else mapping_mlp_params
        self.transformer_params    = {} if transformer_params is None else transformer_params

        self.mapping_params['in_features'], self.mapping_params['out_features'] = self.lattent_size, self.img_size*self.feature_hidden_size
        self.mapping_mlp = MLP(**self.mapping_params)

        self.emb = torch.nn.Parameter(torch.randn(self.img_size, self.feature_hidden_size))

        self.transformer_params['in_features'] = self.feature_hidden_size
        self.transformer_layers = nn.ModuleList([TransformerSLN(**self.transformer_params) for _ in range(self.n_transformer_layers)])

        self.sln = SLN(self.feature_hidden_size)

        self.output_net = nn.Sequential(
            SIREN(self.feature_hidden_size, self.output_hidden_dim, is_first=True),
            SIREN(self.output_hidden_dim, self.n_channels*self.img_size, is_first=False)
        )

        print(f'Generator model with {count_params(self)} parameters ready')

    def forward(self, x):
        w = self.mapping_mlp(x).view(-1, self.img_size, self.feature_hidden_size)
        h = self.emb
        for tf in self.transformer_layers:
            w, h = tf(h, w)
        w = self.sln(h, w)
        res = self.output_net(w).view(x.shape[0], self.n_channels, self.img_size, self.img_size)
        return res


if __name__ == '__main__':
    ipt = torch.randn(10, 1024)
    mod = Generator(1024, 64, 3, 200)
    ret = mod.forward(ipt)
    print(ret.shape)
