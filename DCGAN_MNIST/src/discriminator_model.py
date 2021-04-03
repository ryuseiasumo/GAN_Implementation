import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
torch.manual_seed(1234)


class Discriminator(torch.nn.Module):
    def __init__(self, nch = 1, nch_d = 128):
        super().__init__()
        # 1x28x28 → 1次元のスカラーに変換
        self.cnv1 = nn.Conv2d(nch, nch_d, 4, 2, 1)
        torch.nn.init.xavier_uniform_(self.cnv1.weight) #重みの初期化

        self.cnv2 = nn.Conv2d(nch_d, nch_d*2, 4, 2, 1)
        torch.nn.init.xavier_uniform_(self.cnv2.weight) #重みの初期化
        self.bn2 = nn.BatchNorm2d(nch_d*2)

        self.cnv3 = nn.Conv2d(nch_d*2, nch_d*4, 3, 2, 0)
        torch.nn.init.xavier_uniform_(self.cnv3.weight) #重みの初期化
        self.bn3 = nn.BatchNorm2d(nch_d*4)

        self.cnv4 = nn.Conv2d(nch_d*4, 1, 3, 1, 0)
        torch.nn.init.xavier_uniform_(self.cnv4.weight) #重みの初期化


    def forward(self, img_input):
        x1 = self.cnv1(img_input)
        x1 = F.leaky_relu(x1,0.2, inplace=True)

        x2 = self.bn2(self.cnv2(x1))
        x2 = F.leaky_relu(x2,0.2, inplace=True)

        x3 = self.bn3(self.cnv3(x2))
        x3 = F.leaky_relu(x3,0.2, inplace=True)

        x4 = self.cnv4(x3)
        out = torch.sigmoid(x4)

        return out.squeeze()


from torchsummary import summary
if __name__ == "__main__":
    model = Discriminator()
    summary(model,(1, 28, 28)) # summary(model,(channels,H,W))
