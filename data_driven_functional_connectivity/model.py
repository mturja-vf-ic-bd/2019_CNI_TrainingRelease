import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


class FCModel(nn.Module):
    def __init__(self, n_nodes, n_sig):
        super(FCModel, self).__init__()
        self.n_nodes = n_nodes
        self.n_sig = n_sig
        self.adj_w = Parameter(torch.FloatTensor(n_nodes, n_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.adj_w.size(1))
        self.adj_w.data.uniform_(-stdv, stdv)

    def forward(self, X):
        out = -torch.sum(torch.mul(torch.einsum('ipk, iqk -> ipq', [X, X]), self.adj_w))
        l1_loss = torch.sum(torch.abs(self.adj_w))
        print("dot produce: {}, l1_loss: {}".format(out.data, l1_loss.data))
        return out + l1_loss*5e10


class NodeRNN(nn.Module):
    def __init__(self, n_sig, n_hidden, n_layer, n_nodes,
                 kernel_size):
        super(NodeRNN, self).__init__()
        self.n_sig = n_sig
        self.n_hidden = n_hidden
        self.conv_1 = nn.Conv1d(n_nodes, n_nodes, kernel_size, 1)
        self.conv_2 = nn.Conv1d(n_nodes, n_nodes, kernel_size, 2)
        self.maxpool = nn.MaxPool1d(4, stride=4)
        self.gru = nn.GRU(17, n_hidden, n_layer,
                          bidirectional=True, batch_first=True)
        self.dense = nn.Linear(2 * n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_sig):
        out = self.conv_1(in_sig)
        out = self.conv_2(out)
        out = self.maxpool(out)
        out, h = self.gru(out)
        out = torch.cat((out[:, -1, :self.n_hidden], out[:, 0, self.n_hidden:]), 1)
        out = self.sigmoid(self.dense(out))
        return out

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.gru.reset_parameters()
        self.dense.reset_parameters()
