import argparse
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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


# 加载数据
tmp = np.load("C:\\Users\\MSI\\PycharmProjects\\AI\\DeepConvLSTM_with_selfAttention\\WISDM.npz", allow_pickle=True)

# (20846, 1, 100, 3)
X = tmp['X']
X = np.squeeze(X, axis=1)  # X形状为 (20846, 100, 3)

folds = tmp['folds']  # folds形状为(10, 2)

y_one_hot = tmp['y']  # y_one_hot 形状为 (20846, 6)  -> [1. 0. 0. 0. 0. 0.]
y = np.argmax(y_one_hot, axis=1)

train_idx = folds[0, 0]
test_idx = folds[0, 1]

# (18393, 100, 3)，(18393,)，(18393, 6)
X_train, y_train, y_train_one_hot = X[train_idx], y[train_idx], y_one_hot[train_idx]

# (2453, 100, 3)，(2453,)，(2453, 6)
X_test, y_test, y_test_one_hot = X[test_idx], y[test_idx], y_one_hot[test_idx]

myTrainDataSet = MyDataSet(X_train, y_train)
myTestDataSet = MyDataSet(X_test, y_test)

train_loader = torch.utils.data.DataLoader(myTrainDataSet, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(myTestDataSet, batch_size=args.batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(300, 150)
        self.fc21 = nn.Linear(150, 20)
        self.fc22 = nn.Linear(150, 20)
        self.fc3 = nn.Linear(20, 150)
        self.fc4 = nn.Linear(150, 300)

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
        mu, logvar = self.encode(x.view(-1, 300))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 300), reduction='sum')

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
    max_BCE = 0
    min_BCE = 200
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
    print(
        '====> Test set loss: {:.4f},Test BCE loss: {:.4f},Test KLD loss: {:.4f}'.format(
            test_loss, test_BCE_LOSS,
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
    min_BCE_loss = 300

    for i in range(100):
        bad_input = torch.randn([1, 100, 3])
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
    ====> Test set loss: 138.6301,Test BCE loss: 129.1936,Test KLD loss: 9.4365
    For good input, the reconstruction loss is : 107.77568054199219
    For bad input, the mean reconstruction loss is : 229.1925846862793
    For bad input, the min reconstruction loss is : 215.39723205566406
    For bad input, the max reconstruction loss is : 247.32684326171875
    '''



