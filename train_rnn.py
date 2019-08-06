from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import time
from parser import load_data
from data_driven_functional_connectivity.model import NodeRNN, NodeConv, init_param, SAE
from util import accuracy, pca_dim_red, plot_signal
from data_driven_functional_connectivity.AutoEncoder import EncoderRNN, SigAutoEncoder

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--kernel_size', type=int, default=3,
                    help='kernel size in 1d conv layer')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
data, label, train_idx, test_idx = load_data('aal')
data = pca_dim_red(data, 30)
data = torch.FloatTensor(data).unsqueeze(1)
label = torch.FloatTensor(label).view(-1, 1)
train_idx = torch.LongTensor(train_idx)
test_idx = torch.LongTensor(test_idx)

# Model and optimizer
# model = NodeRNN(data.size(3), [5, 1], args.hidden, 1, 2, args.dropout)
# model = NodeConv([data.size(1), 30, 5, 1], kernel_size=5, dropout=args.dropout, sigmoid=True)
# model = SAE(conv_seq=[data.size(1), 50, 20], conv_kernel=10,
#             deconv_seq=[20, 50, data.size(1)], deconv_kernel=5,
#             deconv_stride=[5, 6], dropout=0.6)
# model = EncoderRNN(data.size(3), [5, 1], args.hidden, 1, 2, args.dropout)
model = SigAutoEncoder(data.size(3), [5, 1], args.kernel_size, args.hidden, args.hidden*2, args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model = model.cuda()
    data = data.cuda()
    torch.cuda.manual_seed(args.seed)
    train_idx = train_idx.cuda()
    test_idx = test_idx.cuda()
    label = label.cuda()


def train(epoch, train_idx, test_idx):
    model.train()
    optimizer.zero_grad()
    input = data[train_idx]
    # input = input[0:10]

    output, mu, logvar = model(input)

    loss_fn = torch.nn.MSELoss()
    loss = 100*loss_fn(output, input) - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # loss_fn = torch.nn.BCELoss()
    # loss = loss_fn(output, label[train_idx])
    # loss.backward()
    optimizer.step()

    plot_signal(output.detach().numpy()[0:3])
    if epoch%5 == 0:
        print('epoch: {},  loss: {}'.format(epoch, loss.data))
    # return tst_rnn(train_idx), tst_rnn(test_idx), loss.data
    return loss.data

    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     model.eval()
    #     output = model(features, adj)

    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))


def tst_rnn(idx_test):
    model.eval()
    output = model(data[idx_test])
    acc = accuracy(output.cpu().data, label[idx_test].cpu().data)
    print("Test set results:",
          "accuracy= {:.4f}".format(acc))
    return acc


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) \
            or isinstance(m, nn.Linear) or isinstance(m, nn.GRU):
        m.reset_parameters()

# Train model
t_total = time.time()
acc = []
from matplotlib import pyplot as plt
acc_batch = np.zeros(args.epochs)
acc_tr = []
for i in range(len(test_idx)):
    model.apply(weight_init)
    acc_train = []
    acc_test = []
    loss_list = []
    for epoch in range(args.epochs):
        # a, b, loss = train(epoch, train_idx[i], test_idx[i])
        loss = train(epoch, train_idx[i], test_idx[i])
        # acc_train.append(a)
        # acc_test.append(b)
        loss_list.append(loss)

    # acc.append(tst_rnn(test_idx[i]))
    # plt.plot(acc_train, "r-")
    # plt.plot(acc_test, "b")
    plt.plot(loss_list)
    plt.show()
    # acc_batch = acc_batch + np.array(acc_test)
# acc_batch /= len(test_idx)
print(loss_list)
# plt.plot(acc_batch)
plt.show()


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("Mean Accuracy: {:.2f}".format(np.mean(acc)))