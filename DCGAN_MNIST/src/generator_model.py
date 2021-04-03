import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
torch.manual_seed(0)

class Generator(torch.nn.Module):
    def __init__(self, z_input_size = 100): #入力ベクトルの次元数は100
        super().__init__()
        #1x1x100次元の入力ベクトルから、64x64x3の画像を生成
        self.cnvt1 = nn.ConvTranspose2d(z_input_size, 256, 1, 1, 0, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt1.weight) #重みの初期化
        self.bn1 = nn.BatchNorm2d(256)

        self.cnvt2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt2.weight) #重みの初期化
        self.bn2 = nn.BatchNorm2d(128)

        self.cnvt3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt3.weight) #重みの初期化
        self.bn3 = nn.BatchNorm2d(64)

        self.cnvt4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt4.weight) #重みの初期化
        self.bn4 = nn.BatchNorm2d(32)

        self.cnvt5 = nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnvt5.weight) #重みの初期化


    def forward(self, z_input):
        x1 = self.bn1(self.cnvt1(z_input))
        x1 = F.leaky_relu(x1, inplace=True)

        x2 = self.bn2(self.cnvt2(x1))
        x2 = F.leaky_relu(x2, inplace=True)

        x3 = self.bn3(self.cnvt3(x2))
        x3 = F.leaky_relu(x3, inplace=True)

        x4 = self.bn4(self.cnvt4(x3))
        x4 = F.leaky_relu(x4, inplace=True)

        x5 = self.cnvt5(x4)
        out = torch.tanh(x5)

        return out


from torchsummary import summary
if __name__ == "__main__":
    model = Generator()
    summary(model,(100, 1, 1)) # summary(model,(channels,H,W))
