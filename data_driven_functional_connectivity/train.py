from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

import torch
import torch.optim as optim
import time
from parser import parse_data_class
from data_driven_functional_connectivity.model import FCModel

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=148,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Load data
data = torch.FloatTensor(parse_data_class('Control', 'aal'))

# Model and optimizer
model = FCModel(n_nodes=data.size(1),
            n_sig=data.size(2))
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    data.cuda()
    torch.cuda.manual_seed(args.seed)


def train(epoch):
    model.train()
    optimizer.zero_grad()

    loss = model(data)
    loss.backward()
    optimizer.step()
    print('epoch: {},  loss: {}'.format(epoch, loss.data))

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


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

from util import plot_matrix, process_matrix
#w = process_matrix(model.adj_w.detach().numpy(), p=50)
w = model.adj_w.detach().numpy()
plot_matrix(w)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))