import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
torch.manual_seed(1234)

class Generator(torch.nn.Module):
    def __init__(self, z_input_size = 100, nch_g = 128, nch = 1): #入力ベクトルの次元数は100
        super().__init__()
        #1x1x100次元の入力ベクトルから、28x28x1の画像を生成
        self.cnvt1 = nn.ConvTranspose2d(z_input_size, nch_g * 4, 3, 1, 0, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt1.weight) #重みの初期化
        self.bn1 = nn.BatchNorm2d(nch_g * 4)

        self.cnvt2 = nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 3, 2, 0, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt2.weight) #重みの初期化
        self.bn2 = nn.BatchNorm2d(nch_g * 2)

        self.cnvt3 = nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt3.weight) #重みの初期化
        self.bn3 = nn.BatchNorm2d(nch_g)

        self.cnvt4 = nn.ConvTranspose2d(nch_g, nch, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt4.weight) #重みの初期化

    def forward(self, z_input):
        x1 = self.bn1(self.cnvt1(z_input))
        x1 = F.relu(x1, inplace=True)

        x2 = self.bn2(self.cnvt2(x1))
        x2 = F.relu(x2, inplace=True)

        x3 = self.bn3(self.cnvt3(x2))
        x3 = F.relu(x3, inplace=True)

        x4 = self.cnvt4(x3)
        out = torch.tanh(x4)

        return out

from torchsummary import summary
if __name__ == "__main__":
    model = Generator()
    summary(model,(100, 1, 1)) # summary(model,(channels,H,W))
