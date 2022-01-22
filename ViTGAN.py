import torch
import torch.nn

from Core.Discriminator import Discriminator
from Core.Generator import Generator
from Core.PytorchGAN import PytorchGAN


class ViTGAN(PytorchGAN):
    def __init__(self, img_size, n_channels, criterion='bce', logger=None, opt='adam', device='cpu', ckpt_save_path=None, tag=''):
        super().__init__(criterion=criterion, logger=logger, opt=opt, device=device, ckpt_save_path=ckpt_save_path, tag=tag)

        # Necessary attributes for PytorchGAN
        self.generator = Generator()
        self.discriminator = Discriminator(img_size, n_channels)
        self.generator_input_shape = (1, 1)  # exemple

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.to(self.device)

    def forward(self, x):
        pass

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, imgs):
        return self.discriminator(imgs)
