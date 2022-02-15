import os
import json
import torch
import logging
import datetime
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from ViTGAN import ViTGAN


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "XP/tag-" + start_time + '_MNIST'  
    writer = SummaryWriter(save_path)

    img_size = 32  # 64
    n_channels = 1  # 3
    lattent_space_size = 1024

    """
    # download celebA
    print('Loading ds')
    # Path for remote use on TME GPU
    path = os.path.abspath('../../../../../tempory/celebA')  # path pour aller le chercher dans tempory depuis home
    dataset = dset.ImageFolder(root=path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(0.5, 0.5, 0.5),
                               ]))
    """

    path = os.path.abspath("../data")
    dataset = dset.MNIST(root=path,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(img_size),
                             transforms.CenterCrop(img_size),
                             transforms.ToTensor(),
                             transforms.Normalize(0.5, 0.5),
                         ]))

    print(dataset)
    seed = 0
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    batch_size = 128
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    generator_params     = {'n_transformer_layers': 4}
    discriminator_params = {'n_transformer_layers': 4}

    model = ViTGAN(img_size=img_size, n_channels=n_channels, generator_params=generator_params, discriminator_params=discriminator_params, lattent_space_size=lattent_space_size, device=device, logger=writer.add_scalar)
    model.fit(dataloader, n_epochs=100, gen_lr=2e-5, disc_lr=2e-5)

    noise = torch.randn(32, lattent_space_size, device=device)
    fake = model.generate(noise)
    img = vutils.make_grid(fake, padding=2, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(save_path, "fake.png"))
