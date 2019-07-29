from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch
import torch.optim as optim
import time
from parser import load_data
from data_driven_functional_connectivity.model import NodeRNN, NodeConv
from util import accuracy

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.007,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--kernel_size', type=int, default=10,
                    help='kernel size in 1d conv layer')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
data, label, train_idx, test_idx = load_data('aal')
data = torch.FloatTensor(data)
label = torch.FloatTensor(label).view(-1, 1)
train_idx = torch.LongTensor(train_idx)
test_idx = torch.LongTensor(test_idx)

# Model and optimizer
# model = NodeRNN(n_sig=data.size(2), n_hidden=args.hidden, n_layer=1, n_nodes=data.size(1), kernel_size=args.kernel_size)
model = NodeConv(n_sig=data.size(2), n1=data.size(1), n2=40, n3=20, n4=10, n5=5, kernel_size=args.kernel_size)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

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

    loss_fn = torch.nn.BCELoss()
    output = model(data[train_idx])
    loss = loss_fn(output, label[train_idx])
    loss.backward()
    optimizer.step()
    if epoch%5 == 0:
        print('epoch: {},  loss: {}'.format(epoch, loss.data))
    return tst_rnn(test_idx)

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


# Train model
t_total = time.time()
acc = []
from matplotlib import pyplot as plt
acc_batch = np.zeros(args.epochs)
for i in range(len(test_idx)):
    model.reset_parameters()
    acc_train = []
    for epoch in range(args.epochs):
        acc_train.append(train(epoch, train_idx[i], test_idx[i]))
    tst_rnn(train_idx[i])
    acc.append(tst_rnn(test_idx[i]))
    acc_batch = acc_batch + np.array(acc_train)
    plt.plot(acc_train)
    plt.show()
acc_batch /= len(test_idx)
print(acc_batch)
plt.plot(acc_batch)
plt.show()


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print("Mean Accuracy: {:.2f}".format(np.mean(acc)))