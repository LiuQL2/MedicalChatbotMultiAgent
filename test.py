from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

K=10 # number of classes
N=30 # number of categorical distributions
tau0 = 1.0
tau = tau0
tau = Variable(torch.tensor(tau))
def sample_gumbel(shape, eps=1e-20):
    U = torch.Tensor(shape).uniform_(0,1).cuda()
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y = (y_hard - y).detach() + y
    return y

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fce1 = nn.Linear(784, 512)
        self.fce2 = nn.Linear(512, 256)
        self.fce3 = nn.Linear(256, K*N)
        self.fcd1 = nn.Linear(K*N, 256)
        self.fcd2 = nn.Linear(256, 512)
        self.fcd3 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def encode(self, x):
        he1 = self.relu(self.fce1(x))
        he2 = self.relu(self.fce2(he1))
        he3 = self.fce3(he2)
        logits_y = he3.view(-1, K)
        qy = self.softmax(logits_y)
        log_qy = torch.log(qy + 1e-20)
        return logits_y, log_qy, qy

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        # sample and reshape back (shape=(batch_size,N,K))
        # set hard=True for ST Gumbel-Softmax
        ge = gumbel_softmax(z, tau, hard=False).view(-1, N, K)
        hd1 = self.relu(self.fcd1(ge.view(-1, N*K)))
        hd2 = self.relu(self.fcd2(hd1))
        hd3 = self.fcd3(hd2)
        return hd3

    def forward(self, x):
        logits_y, log_qy, qy = self.encode(x.view(-1, 784))
        return self.decode(logits_y),log_qy, qy


model = VAE()
if args.cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, log_qy, qy, data):
    sigmoid = nn.Sigmoid()
    kl_tmp = (qy * (log_qy - torch.log(torch.tensor(1.0 / K)))).view(-1, N, K)
    KL = torch.sum(torch.sum(kl_tmp, 2),1)
    shape = data.size()
    #elbo = torch.sum(recon_x.log_prob(data.view(shape[0], shape[1] * shape[2] * shape[3])), 1) - KL
    data_ = data.view(shape[0], shape[1] * shape[2] * shape[3])
    # calculate binary cross entropy using explicit calculation rather than using pytorch distribution API
    bce = torch.sum(data_ * torch.log(sigmoid(recon_x)) + (1 - data_) * torch.log(1 - sigmoid(recon_x)), 1)
    elbo = bce - KL
    return torch.mean(-elbo), torch.mean(bce), torch.mean(KL)

ANNEAL_RATE=0.00003
MIN_TEMP=0.5

def train(epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        px, log_qy, qy = model(data)
        recon_x = torch.distributions.bernoulli.Bernoulli(logits=px)
        #loss = loss_function(recon_x, log_qy, qy, data)
        loss, bce, KL = loss_function(px, log_qy, qy, data)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        #if batch_idx % 1000 == 1:
        #    tau = Variable(torch.tensor(np.maximum(tau0 * np.exp(-ANNEAL_RATE * batch_idx), MIN_TEMP)))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tBCE: {:.6f} \tKL: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0], bce.data[0], KL.data[0]))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    M = 100 * N
    np_y = np.zeros((M, K))
    np_y[range(M), np.random.choice(K, M)] = 1
    np_y = np.reshape(np_y, [100, N, K])

    px = model.decode(Variable(torch.tensor(np_y).cuda()))
    recon_x = torch.nn.Sigmoid()(px).detach().cpu().numpy()
    #recon_x = torch.distributions.Bernoulli(logits=px).sample()
    np_x = recon_x.reshape((10, 10, 28, 28))
    # split into 10 (1,10,28,28) images, concat along columns -> 1,10,28,280
    np_x = np.concatenate(np.split(np_x, 10, axis=0), axis=3)
    # split into 10 (1,1,28,280) images, concat along rows -> 1,1,280,280
    np_x = np.concatenate(np.split(np_x, 10, axis=1), axis=2)
    x_img = np.squeeze(np_x)
    plt.imshow(x_img, cmap=plt.cm.gray, interpolation='none')
    plt.show()

args.epochs = 1
for epoch in range(1, args.epochs + 1):
    train(epoch)