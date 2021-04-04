import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tqdm import trange, tqdm


from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets

from statistics import mean


#シード値
torch.manual_seed(0)

class Trainer:
    def __init__(self, Generator, G_optimizer, Discriminator, D_optimizer, loss_function, num_epochs, batch_size = 50, z_input_size = 100):
        super().__init__()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.model_G = Generator.to(self.device)
        self.model_D = Discriminator.to(self.device)

        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self.z_input_size = z_input_size #潜在特徴100次元ベクトルz
        self.batch_size = batch_size
        self.ones = torch.ones(batch_size).reshape(batch_size, 1).to(self.device) # 正例 1
        self.zeros = torch.zeros(batch_size).reshape(batch_size, 1).to(self.device) # 負例 0

        self.loss_function = loss_function.to(self.device)
        self.num_epochs = num_epochs

        # 途中結果の確認用の潜在変数z
        check_z = torch.randn(batch_size, self.z_input_size, 1, 1).to(self.device)
        check_label = [ i for i in range(10)] * (batch_size // 10)
        check_label += [ j for j in range(batch_size % 10)]

        check_label = torch.tensor(check_label, dtype = torch.long, device = self.device)

        self.check_z_label = self.concat_noise_label(check_z, check_label)



    def onehot_encode(self, label, n_class = 10):
        eye = torch.eye(n_class, device = self.device)
        return eye[label].view(-1, n_class, 1, 1)

    def concat_image_label(self, image, label, n_class = 10):
        batch_size, c, h, w = image.shape #画像Tensorのサイズ情報を取得
        oh_label = self.onehot_encode(label, n_class = n_class)
        oh_label = oh_label.expand(batch_size, n_class, h, w) #クラスラベルを条件画像に変換
        return torch.cat((image, oh_label), dim = 1) #画像に条件画像を連結

    def concat_noise_label(self, z_input, label, n_class = 10):
        oh_label = self.onehot_encode(label, n_class = n_class)
        return torch.cat((z_input, oh_label), dim = 1) #画像生成をするためのノイズ(サンプル)にクラスラベルを連結


    #訓練用
    def train(self, train_loader):
        self.model_G.train()
        self.model_D.train()

        log_loss_G = []
        log_loss_D = []

        l = len(train_loader)

        for i, data in tqdm(enumerate(train_loader),total = l):
            real_img = data[0].to(self.device) #本物画像
            real_label = data[1].to(self.device) #ラベル
            batch_len = real_img.size(0)

            #本物画像とラベルを連結
            real_img_label = self.concat_image_label(real_img, real_label)

            if torch.cuda.is_available():
                real_img_label = Variable(real_img_label.cuda(), volatile=True)
            else:
                real_img_label = Variable(real_img_label, volatile=True)


            # ========== Generatorの訓練 ==========
            # 偽画像を生成
            z = torch.randn(batch_len, self.z_input_size, 1, 1).to(self.device) #バッチサイズ分だけ、標準正規分布から潜在変数を用意
            #偽画像生成用のラベル
            fake_label = torch.randint(10, (batch_len, ), dtype=torch.long, device=self.device)
            #生成用ノイズとクラスラベルを連結
            fake_noize_label = self.concat_noise_label(z, fake_label)

            fake_img = self.model_G(fake_noize_label) #偽画像生成

            #生成画像とクラスラベル(条件画像)を連結
            fake_img_label = self.concat_image_label(fake_img, fake_label)
            # 偽画像の値を一時的に保存
            fake_img_tensor = fake_img_label.detach()

            # 偽画像を実画像（ラベル１）と騙せるようにロスを計算
            out = self.model_D(fake_img_label).reshape(batch_len, -1)
            loss_G = self.loss_function(out, self.ones[: batch_len]) #偽画像(fake_img_label)に対して、間違えるほど良い→Discriminatorが1を出力する程よい
            log_loss_G.append(loss_G.item())

             # 微分計算・重み更新
            self.model_D.zero_grad()
            self.model_G.zero_grad()
            loss_G.backward()
            self.G_optimizer.step()


            # ========== Discriminatorの訓練 ==========
            # 実画像を実画像（ラベル１）と識別できるようにロスを計算
            real_img_label = real_img_label.to(self.device) # sample_dataの実画像を用意
            real_out = self.model_D(real_img_label).reshape(batch_len, -1)
            loss_D_real = self.loss_function(real_out, self.ones[: batch_len]) #正しい画像(real_img)に対して、正しいほど良い→Discriminatorが1を出力する程よい

             # 偽画像を偽画像（ラベル０）と識別できるようにロスを計算
            fake_img_label = fake_img_tensor #偽画像(+条件画像)を用意
            fake_out = self.model_D(fake_img_label).reshape(batch_len, -1)
            loss_D_fake = self.loss_function(fake_out, self.zeros[: batch_len])

            # 実画像と偽画像のロスを合計
            loss_D = loss_D_real + loss_D_fake
            log_loss_D.append(loss_D.item())

            self.model_D.zero_grad()
            self.model_G.zero_grad()
            loss_D.backward()
            self.D_optimizer.step()

        return mean(log_loss_G), mean(log_loss_D)


    def fit_model(self, train_dataloader): #実際にモデルの学習を行う関数(early_stoppingあり)
        if not os.path.exists('Weight_Generator'):
            os.mkdir('Weight_Generator')
        if not os.path.exists('Weight_Discriminator'):
            os.mkdir('Weight_Discriminator')
        if not os.path.exists('Generated_Image'):
            os.mkdir('Generated_Image')
        # Training
        for epoch in trange(self.num_epochs):
            log_loss_G, log_loss_D = self.train(train_dataloader)
            print(f'epoch: {epoch}, netG loss: {log_loss_G}, netD loss: {log_loss_D}')

            # 訓練途中のモデル・生成画像の保存
            if epoch % 10 == 0:
                torch.save(
                    self.model_G.state_dict(),
                    "./Weight_Generator/G_{:03d}.pth".format(epoch),
                    pickle_protocol=4)
                torch.save(
                    self.model_D.state_dict(),
                    "./Weight_Discriminator/D_{:03d}.pth".format(epoch),
                    pickle_protocol=4)

                generated_img = self.model_G(self.check_z_label)
                save_image(generated_img,
                        "./Generated_Image/{:03d}.jpg".format(epoch))
