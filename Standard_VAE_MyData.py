import argparse
import torch
import random
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


class MyDataSet(Dataset):
    def __init__(self, data_root, data_label):
        """
        这里传入的一个是data数据一个是label数据，在行维度上要对应。
        :param data_root:  5 x 3维
        :param data_label:  5 x 1维
        """
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)


# 读数据
filePath = 'E:\\11-16.csv'

# (4835, 1201)
pre_data = np.loadtxt(filePath, dtype=np.float, delimiter=',')


# (4835, 400, 3)
X = np.zeros([pre_data.shape[0], 400, 3])

# (4835, )
y = np.zeros(pre_data.shape[0])

# 填入加速度X,Y,Z轴
for i in range(pre_data.shape[0]):
    for j in range(400):
        X[i, j, 0] = pre_data[i, j + 1]
        X[i, j, 1] = pre_data[i, j + 401]
        X[i, j, 2] = pre_data[i, j + 801]

# 填入标签label
for i in range(pre_data.shape[0]):
    y[i] = pre_data[i, 0] - 1

# 划分训练集 测试集
random_idx = random.sample(range(0, 4835), 4835)
train_idx = random_idx[0:4352]
test_idx = random_idx[4352:4835]

train_data = X[train_idx]  # (4352, 1, 400, 3)
train_label = y[train_idx]  # (4352,)
test_data = X[test_idx]  # (480, 1, 400, 3)
test_label = y[test_idx]  # (480,)

# 定义自己的dataset
myTrainDataSet = MyDataSet(train_data, train_label)
myTestDataSet = MyDataSet(test_data, test_label)

# 定义自己的dataloader
train_loader = torch.utils.data.DataLoader(dataset=myTrainDataSet, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=myTestDataSet, batch_size=args.batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(1200, 600)
        self.fc21 = nn.Linear(600, 20)
        self.fc22 = nn.Linear(600, 20)
        self.fc3 = nn.Linear(20, 600)
        self.fc4 = nn.Linear(600, 1200)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1200))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1200), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.float()
        # [128, 1, 100, 3]
        data = data.to(device)
        data = F.sigmoid(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, BCE_LOSS, KLD_LOSS = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},BCE_LOSS: {:.6f},KLD_LOSS: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data), BCE_LOSS.item() / len(data), KLD_LOSS.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    test_BCE_LOSS = 0
    test_KLD_LOSS = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.float()
            data = data.to(device)
            data = F.sigmoid(data)
            recon_batch, mu, logvar = model(data)
            loss, BCE_LOSS, KLD_LOSS = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            test_BCE_LOSS += BCE_LOSS.item()
            test_KLD_LOSS += KLD_LOSS.item()

    test_loss /= len(test_loader.dataset)
    test_BCE_LOSS /= len(test_loader.dataset)
    test_KLD_LOSS /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f},Test BCE loss: {:.4f},Test KLD loss: {:.4f}'.format(test_loss, test_BCE_LOSS,
                                                                                           test_KLD_LOSS))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

    # 挑选一个好的input
    d, l = next(iter(test_loader))
    good_input = d[3, :, :]
    good_input = good_input.unsqueeze(dim=0)
    good_input = good_input.float()
    good_input = F.sigmoid(good_input)
    good_input = good_input.to(device)
    recon_input, good_mu, good_logvar = model(good_input)
    good_loss, good_BCE_LOSS, good_KLD_LOSS = loss_function(recon_input, good_input, good_mu, good_logvar)
    print("For good input, the reconstruction loss is : " + str(good_BCE_LOSS.item()))


    # 挑选一个不好的
    mean_BCE_loss = 0
    max_BCE_loss = 0
    min_BCE_loss = 1000

    for i in range(100):
        bad_input = torch.randn([1, 400, 3])
        bad_input = bad_input.float()
        bad_input = F.sigmoid(bad_input)
        bad_input = bad_input.to(device)
        recon_bad, bad_mu, bad_logvar = model(bad_input)
        bad_loss, bad_BCE_LOSS, bad_KLD_LOSS = loss_function(recon_bad, bad_input, bad_mu, bad_logvar)
        mean_BCE_loss+=bad_BCE_LOSS.item()
        if bad_BCE_LOSS.item() >= max_BCE_loss:
            max_BCE_loss = bad_BCE_LOSS.item()
        if bad_BCE_LOSS.item() <= min_BCE_loss:
            min_BCE_loss = bad_BCE_LOSS.item()


    print("For bad input, the mean reconstruction loss is : " + str(mean_BCE_loss/100))
    print("For bad input, the min reconstruction loss is : " + str(min_BCE_loss))
    print("For bad input, the max reconstruction loss is : " + str(max_BCE_loss))


    '''
    ====> Test set loss: 813.9601,Test BCE loss: 807.0527,Test KLD loss: 6.9075
    For good input, the reconstruction loss is : 745.7249755859375
    For bad input, the mean reconstruction loss is : 834.9137322998047
    For bad input, the min reconstruction loss is : 831.8546752929688
    For bad input, the max reconstruction loss is : 842.0533447265625
    '''
