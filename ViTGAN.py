import torch
import torch.nn

from Core.Discriminator import Discriminator
from Core.Generator import Generator
from Core.PytorchGAN import PytorchGAN


class ViTGAN(PytorchGAN):
    def __init__(self, img_size, n_channels, lattent_space_size, generator_params=None, discriminator_params=None, criterion='bce', logger=None, opt='adam', device='cpu', ckpt_save_path=None, tag='', **kwargs):
        """
        Main VitGAN class for this project
        :param img_size: images size, the image must be square sized
        :param n_channels: number of channel of the images
        :param lattent_space_size: umber of features in the lattent space
        :param generator_params: kwargs for optional parameters of the Generator, mandatory args will be filled automatically
        :param discriminator_params: kwargs for optional parameters of the Discriminator, mandatory args will be filled automatically
        :param criterion: loss used for training, BCE or MSE
        :param logger: tensorboard logger
        :param opt: optimizer to use for training
        :param device: cpu or cuda
        :param ckpt_save_path: save path for training checkpoints
        :param tag: model tag for saved file names
        """
        super().__init__(criterion=criterion, logger=logger, opt=opt, device=device, ckpt_save_path=ckpt_save_path, tag=tag)

        self.img_size           = img_size
        self.n_channels         = n_channels
        self.lattent_space_size = lattent_space_size

        self.generator_params       = {} if generator_params is None else generator_params
        self.discriminator_params   = {} if discriminator_params is None else discriminator_params

        self.generator_params['img_size'], self.generator_params['n_channels'], self.generator_params['lattent_space_size'] = self.img_size, self.n_channels, self.lattent_space_size
        self.discriminator_params['img_size'], self.discriminator_params['n_channels'], self.discriminator_params['output_size'] = self.img_size, self.n_channels, 2

        # Necessary attributes for PytorchGAN
        self.generator = Generator(**self.generator_params)
        self.discriminator = Discriminator(**self.discriminator_params)
        self.generator_input_shape = (self.lattent_space_size,)  # exemple

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.to(self.device)

    def forward(self, x):
        pass

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, imgs):
        return self.discriminator(imgs)
