import argparse
import os
import time
import cnnf_torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--cnnf_weight', type=str,
                    default='/home/aistudio/data/data20371/vgg_net.mat',
                    help="CNN-F weights file path")
parser.add_argument('--data_path', type=str, default="E:/iTom/dataset/")
parser.add_argument('--log_path', type=str, default="log")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_class', type=int, default=10)
parser.add_argument('--epoch', type=int, default=14)
args = parser.parse_args()


def timestamp():
    """time-stamp string: Y-M-D-h-m"""
    t = time.localtime(time.time())
    return "{}-{}-{}-{}-{}".format(
        t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)


if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
log_file_path = os.path.join(args.log_path, "log.{}".format(timestamp()))
log_file = open(log_file_path, "a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, optimizer, criterion):
    model.train()
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()


def test(model, loader):
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            out = model(X)
            pred = out.argmax(dim=1)
            n_correct += pred.eq(Y.view_as(pred)).sum().item()
    acc = n_correct / len(loader.dataset)
    return acc


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_path, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize([224, 224]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                   ])),
    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_path, train=False,
                   transform=transforms.Compose([
                       transforms.Resize([224, 224]),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                   ])),
    batch_size=args.batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self, n_class):
        super(Net, self).__init__()
        self.cnnf = cnnf_torch.CNN_F(args.cnnf_weight)
        self.fc = nn.Linear(4096, args.n_class)

    def forward(self, x):
        x = self.cnnf(x)
        logit = self.fc(x)
        return logit


model = Net(args.n_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# for x, y in train_loader:
#     print(x.size(), y.size())  # [n, 1, 28, 28], [64]
#     break
for epoch in range(args.epoch):
    print("---", epoch, "---")
    train(model, train_loader, optimizer, criterion)
    acc = test(model, test_loader)
    print("acc:", acc)
    log_file.write("acc: {}\n".format(acc))


log_file.flush()
log_file.close()
