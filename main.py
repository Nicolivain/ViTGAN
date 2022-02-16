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

    with open('config.json', 'rb') as f:
        config = json.load(f)

    config['img_size']          = 32
    config['n_channels']        = 1

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    config['logger']            = writer
    config['ckpt_save_path']    = None

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
    torch.manual_seed(config['seed'])
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', config['device'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)

    model = ViTGAN(**config)
    model.fit(dataloader, n_epochs=100, gen_lr=2e-5, disc_lr=2e-5, save_images_freq=1)

    noise = torch.randn(32, config['lattent_space_size'], device=config['device'])
    fake = model.generate(noise)
    img = vutils.make_grid(fake, padding=2, normalize=True)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(save_path, "fake.png"))
