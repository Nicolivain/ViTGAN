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

    with open('config.json', "rb") as f:
        config = json.load(f)

    config['img_size']              = 32  # 64
    config['n_channels']            = 1  # 3
    config['lattent_space_size']    = 1024

    """
    # CelebA
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
                             transforms.Resize(config['img_size']),
                             transforms.CenterCrop(config['img_size']),
                             transforms.ToTensor(),
                             transforms.Normalize(0.5, 0.5),
                         ]))
    print(dataset)

    config['seed'] = 0
    torch.manual_seed(config['seed'])
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', config['device'])

    config['batch_size'] = 128
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)

    config['gen_lr']  = 2e-5
    config['disc_lr'] = 2e-5

    # save the config with the logs
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    config['logger'] = writer

    model = ViTGAN(**config)
    model.fit(dataloader, n_epochs=100, gen_lr=config['gen_lr'], disc_lr=config['gen_lr'])

    # Save a sample of generated images at the end of training
    noise = torch.randn(32, config['lattent_space_size'], device=config['device'])
    fake = model.generate(noise)
    img = vutils.make_grid(fake, padding=2, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(save_path, "fake.png"))


