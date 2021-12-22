import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils import data
from all_layer import DeepConvLSTM_with_selfAttention
from visdom import Visdom
import random
from VAE import VAE, loss_fn
import torch.nn.functional as func

EPOCH = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

viz = Visdom()
viz.line([0.], [0.], win="all_loss", opts=dict(title="all_loss"))
viz.line([0.], [0.], win="BCE_loss", opts=dict(title="BCE_loss"))
viz.line([0.], [0.], win="KLD_loss", opts=dict(title="KLD_loss"))


class MyDataSet(Dataset):
    def __init__(self, data_root, data_label):
        """
        这里传入的一个是data数据一个是label数据，在行维度上要对应。
        :param data_root:
        :param data_label:
        """
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # 读数据
    filePath = 'E:\\new_320_all_feature.csv'
    # (4835, 322)
    feature_data = np.loadtxt(filePath, dtype=np.float, delimiter=',', skiprows=1)
    # (4835, 321)
    feature_data = feature_data[:, 1:]

    # label
    y = np.zeros(feature_data.shape[0])

    # 将标签装入y
    for i in range(feature_data.shape[0]):
        y[i] = feature_data[i, 0]

    # (4835, 320)
    X = feature_data[:, 1:]

    # 定义dataset
    myDataset = MyDataSet(X, y)

    # 定义dataloader
    myLoader = data.DataLoader(dataset=myDataset, batch_size=BATCH_SIZE, shuffle=True)

    # 定义模型
    vae = VAE(input_size=320, h_dim=128, z_dim=32)
    vae = vae.to(device)

    # 定义损失函数和优化方法
    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        train_loss = 0
        train_BCE_LOSS = 0
        train_KLD_LOSS = 0
        vae.train()
        for step, (sensor, label) in enumerate(myLoader):
            sensor = sensor.float()
            sensor = func.sigmoid(sensor)
            sensor = sensor.to(device)
            label = label.long()
            label = label.to(device)

            recon_x, mu, logvar = vae(sensor)

            all_loss, BCE_loss, KLD_loss = loss_fn(recon_x, sensor, mu, logvar)

            train_loss+= all_loss.item()
            train_BCE_LOSS+=BCE_loss.item()
            train_KLD_LOSS+=KLD_loss.item()

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            # 每100个step，输出一下训练的情况
            if step % 100 == 0:
                print("train epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}".
                      format(epoch,
                             step * len(sensor),
                             len(myLoader.dataset),
                             100. * step / len(myLoader),
                             all_loss.item() / len(sensor)))

        # 完成一个epoch，看一下loss
        print(
            "\t===============epoch {} done, the all_loss is {:.5f},the BCE_loss is {:.5f},the KLD_loss is {:.5f}===============\t".format(
                epoch, train_loss/len(myLoader.dataset), train_BCE_LOSS/len(myLoader.dataset), train_KLD_LOSS/len(myLoader.dataset)))
        viz.line([all_loss.item()], [epoch], win="all_loss", update="append")
        viz.line([BCE_loss.item()], [epoch], win="BCE_loss", update="append")
        viz.line([KLD_loss.item()], [epoch], win="KLD_loss", update="append")


    # 测试一下
    batch_data,batch_label = next(iter(myLoader))
    good_input = batch_data[0,:]
    good_input = good_input.unsqueeze(dim=0)
    good_input = func.sigmoid(good_input)
    good_input = good_input.float()
    good_input = good_input.to(device)

    bad_input = torch.randn([1,320])
    bad_input = func.sigmoid(bad_input)
    bad_input = bad_input.float()
    bad_input = bad_input.to(device)

    good_recon, good_mu, good_logvar = vae(good_input)
    bad_recon, bad_mu, bad_logvar = vae(bad_input)

    good_all_loss, good_BCE_loss, good_KLD_loss = loss_fn(good_recon, good_input, good_mu, good_logvar)
    bad_all_loss, bad_BCE_loss, bad_KLD_loss = loss_fn(bad_recon, bad_input, bad_mu, bad_logvar)



    print("For good input, the reconstruction loss is : " + str(good_BCE_loss.item()))
    print("For bad input, the reconstruction loss is : " + str(bad_BCE_loss.item()))




