import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
torch.manual_seed(0)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 3x64x64 → 1次元のスカラーに変換
        self.cnv1 = nn.Conv2d(1, 32, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnv1.weight) #重みの初期化

        self.cnv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnv2.weight) #重みの初期化
        self.bn2 = nn.BatchNorm2d(64)

        self.cnv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnv3.weight) #重みの初期化
        self.bn3 = nn.BatchNorm2d(128)

        self.cnv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.cnv4.weight) #重みの初期化
        self.bn4 = nn.BatchNorm2d(256)

        self.cnv5 = nn.Conv2d(256, 1, 1, 1, 0, bias=False) #fcの代わり 1x1x1になる
        torch.nn.init.xavier_uniform_(self.cnv5.weight) #重みの初期化


    def forward(self, img_input):
        x1 = self.cnv1(img_input)
        x1 = F.leaky_relu(x1,0.2, inplace=True)

        x2 = self.bn2(self.cnv2(x1))
        x2 = F.leaky_relu(x2,0.2, inplace=True)

        x3 = self.bn3(self.cnv3(x2))
        x3 = F.leaky_relu(x3,0.2, inplace=True)

        x4 = self.bn4(self.cnv4(x3))
        x4 = F.leaky_relu(x4,0.2, inplace=True)

        x5 = self.cnv5(x4)
        out = torch.sigmoid(x5)

        return out





from torchsummary import summary
if __name__ == "__main__":
    model = Discriminator()
    summary(model,(1, 28, 28)) # summary(model,(channels,H,W))
