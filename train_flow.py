
import dnnlib
from torch import nn, optim
import torch
import numpy as np
from torch.utils import data
from module.flow import cnf
from math import log, pi
import os
from tqdm import tqdm

import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as dset
import argparse



def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


class MyDataset(Dataset):
    def __init__(self, latents, attributes, transform=None):
        self.latents = latents
        self.attributes = attributes
        self.transform = transform

    def __getitem__(self, index):
        x = self.latents[index]
        y = self.attributes[index]


        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.latents)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="StyleFlow trainer")

    parser.add_argument("--latent_path",default='data_numpy/latents.npy', type=str, help="path to the latents")
    parser.add_argument("--light_path",default='data_numpy/lighting.npy', type=str, help="path to the lighting parameters")
    parser.add_argument("--attributes_path",default='data_numpy/attributes.npy', type=str, help="path to the attribute parameters")
    parser.add_argument(
        "--batch", type=int, default=5, help="batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs"
    )

    parser.add_argument("--flow_modules", type=str, default='512-512-512-512-512')
    parser.add_argument("--cond_size", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-3)


    args = parser.parse_args()
    torch.manual_seed(0)

    prior = cnf(512, args.flow_modules, args.cond_size, 1)

    sg_latents = np.load(args.latent_path)
    lighting = np.load(args.light_path)
    attributes = np.load(args.attributes_path)
    sg_attributes = np.concatenate([lighting,attributes], axis = 1)

    my_dataset = MyDataset(latents=torch.Tensor(sg_latents).cuda(), attributes=torch.tensor(sg_attributes).float().cuda())
    train_loader = data.DataLoader(my_dataset, shuffle=False, batch_size=args.batch)

    optimizer = optim.Adam(prior.parameters(), lr=args.lr)
   

    with tqdm(range(args.epochs)) as pbar:
        for epoch in pbar:
            for i, x in enumerate(train_loader):
              
                approx21, delta_log_p2 = prior(x[0].squeeze(1), x[1], torch.zeros(args.batch, x[0].shape[2], 1).to(x[0]))

                approx2 = standard_normal_logprob(approx21).view(args.batch, -1).sum(1, keepdim=True)

              
                delta_log_p2 = delta_log_p2.view(args.batch, x[0].shape[2], 1).sum(1)
                log_p2 = (approx2 - delta_log_p2)

                loss = -log_p2.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(
                    f'logP: {loss:.5f}')

                if i % 1000 == 0:
                    torch.save(
                        prior.state_dict(), f'trained_model/modellarge10k_{str(i).zfill(6)}_{str(epoch).zfill(2)}.pt'
                    )

