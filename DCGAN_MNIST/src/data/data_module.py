import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold
from torchvision.datasets import MNIST
from pathlib import Path
import sys, os




# from make_expanded import generate_tokenizer

class Mnist_DataModule:
    def __init__(self, transform=None): #transformが特になければ、ただそのまま返す関数(lambda x:x)とする.
        super().__init__()
        if transform != None:
            self.transform = transform
        else:
            self.transform = self.Mnist_transform()
        self.dataset = MNIST('.', train=True, transform=self.transform, download=True)

    def Mnist_transform(self): #前処理用関数
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

    def train_dataloader(self, batch_size) -> DataLoader:
        return DataLoader(self.dataset , batch_size=batch_size, shuffle=True)
