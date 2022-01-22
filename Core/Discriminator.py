import torch
import torch.nn as nn

from Components.PatchEncoder import PatchEncoder
from Components.Tranformer import Transformer
from Components.MLP import MLP

from Tools.utils import count_params


class Discriminator(nn.Module):
    def __init__(self, img_size, n_channels, output_size, n_transformer_layers=1, encoder_params=None, transformer_params=None, mlp_params=None, **kwargs):
        """
        Discriminator module for ViTGAN model
        :param img_size: input images size, the image must be square sized
        :param n_channels: number of channel in the input images
        :param output_size: number of output features per input image
        :param n_transformer_layers: number of stacked transformer blocks
        :param encoder_params: kwargs for optional parameters of the PatchEncoder, mandatory args will be filled automatically
        :param transformer_params: kwargs for optional parameters for each Transformer block, mandatory args will be filled automatically
        :param mlp_params: kwargs for optional parameters of the output MLP module, mandatory args will be filled automatically
        """
        super(Discriminator, self).__init__()

        self.img_size             = img_size
        self.n_channels           = n_channels
        self.output_size          = output_size
        self.n_transformer_layers = n_transformer_layers

        self.encoder_params     = {} if encoder_params is None else encoder_params
        self.transformer_params = {} if transformer_params is None else transformer_params
        self.mlp_params         = {} if mlp_params is None else mlp_params

        self.encoder_params['img_size'], self.encoder_params['n_channels'] = self.img_size, self.n_channels
        self.patch_encoder = PatchEncoder(**self.encoder_params)

        self.transformer_params['in_features'] = self.patch_encoder.token_size
        self.transformer_layers = nn.ModuleList([Transformer(**self.transformer_params) for _ in range(self.n_transformer_layers)])

        self.mlp_params['in_features'], self.mlp_params['out_features'] = self.transformer_layers[-1].in_features, self.output_size
        self.mlp = MLP(**self.mlp_params)

        self.sigmoid = torch.nn.Sigmoid()

        print(f'Discriminator model with {count_params(self)} parameters ready')

    def forward(self, imgs):
        tokens = self.patch_encoder(imgs)
        for transformer in self.transformer_layers:
            tokens = transformer(tokens)
        output = self.mlp(tokens[:, 0, :])  # we compute the output only with the cls token
        return self.sigmoid(output)


if __name__ == '__main__':
    x = torch.randn(100, 3, 64, 64)
    d = Discriminator(64, 3, 2, 2)
    o = d(x)
    print(o.shape)
