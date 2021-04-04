#mnistを用いた、手書き数字画像生成
#生成する数字を指定する

import torch
from torch import optim
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F
from tqdm import trange, tqdm

import argparse

torch.manual_seed(0)

from src.data import Mnist_DataModule
from src import generator_model
from src import discriminator_model
from src.train import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size', default=64, type=int)
    parser.add_argument('--max_epoch', dest='max_epoch', help='Num of max epoch', default=101, type=int)

    args = parser.parse_args()
    return args

args = parse_args()
num_epochs = args.max_epoch
batch_size = args.batch_size


def set_model(): #モデル等の用意
    Generator = generator_model.Generator()
    Discriminator = discriminator_model.Discriminator()

    G_optimizer  = torch.optim.AdamW(Generator.parameters(), lr=0.001)
    D_optimizer  = torch.optim.AdamW(Discriminator.parameters(), lr=0.001)
    # G_optimizer  = torch.optim.AdamW(Generator.parameters(), lr=0.001, betas = (beta1,0.999), weight_decay = 1e-5)
    # D_optimizer  = torch.optim.AdamW(Discriminator.parameters(), lr=0.001, betas = (beta1,0.999), weight_decay = 1e-5)

    loss_function = nn.BCELoss() #バイナリクロスエントロピーloss

    return Generator, G_optimizer, Discriminator, D_optimizer, loss_function



if __name__ == "__main__":

    #モデルの用意
    Generator, G_optimizer, Discriminator, D_optimizer, loss_function = set_model()
    print(Generator)
    print(Discriminator)

    # データセットの用意
    Mnist_Data = Mnist_DataModule()
    # dataloaderの作成
    train_dataloader = Mnist_Data.train_dataloader(batch_size)

    # import pdb; pdb.set_trace()

    #モデルの学習
    trainer = Trainer(Generator, G_optimizer, Discriminator, D_optimizer, loss_function, num_epochs, batch_size)
    trainer.fit_model(train_dataloader)

    print("Finished !!")



# from src.out_prediction import Out_prediction


# from torchvision.utils import save_image
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torchvision import transforms, datasets

# from statistics import mean

# from pathlib import Path
# SWD = Path(__file__).resolve().parent
