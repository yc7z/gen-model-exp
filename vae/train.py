
import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T

from vae import *

import argparse

from vae.vae import VariationalAutoencoderCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=30)
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # dataset and dataloaders
    mnist = MNIST(root="dataset", train=False, transform=T.ToTensor(), download=True)
    mnist_loader = DataLoader(dataset=mnist, batch_size=args.batch_size, shuffle=True)
    
    # model
    vae = VariationalAutoencoderCNN(args.latent_dim).to(device)
    
    # optimizer
    optimizer = SGD(vae.parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # for epoch in range(1, args.epochs+1):
        
