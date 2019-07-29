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


class NodeConv(nn.Module):
    def __init__(self, n_sig, n1, n2, n3, n4, n5,
                 kernel_size):
        super(NodeConv, self).__init__()
        self.n_sig = n_sig
        self.conv_1 = nn.Conv1d(n1, n2, kernel_size, 1)
        self.conv_2 = nn.Conv1d(n2, n3, kernel_size, 4)
        self.conv_bn_1 = nn.BatchNorm1d(n3)
        self.conv_3 = nn.Conv1d(n3, n4, kernel_size, 3)
        self.conv_4 = nn.Conv1d(n4, 1, 9, 1)
        self.conv_bn_2 = nn.BatchNorm1d(n4)
        self.conv_5 = nn.Conv1d(n5, 1, 1, 1)
        self.drop = nn.Dropout()
        self.linear = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_sig):
        out = self.conv_1(in_sig)
        out = self.drop(out)
        out = self.conv_2(out)
        out = self.conv_bn_1(out)
        out = self.drop(out)
        out = self.conv_3(out)
        out = self.conv_bn_2(out)
        out = self.drop(out)
        out = self.conv_4(out)
        out = self.drop(out)
        #out = self.conv_5(out)
        # out = self.linear(out)
        out = out.view(-1, 1)
        out = self.sigmoid(out)
        return out

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.conv_5.reset_parameters()
        self.conv_bn_1.reset_parameters()
        self.conv_bn_2.reset_parameters()


class NodeRNN(nn.Module):
    def __init__(self, n_sig, n_hidden, n_layer, n_nodes,
                 kernel_size):
        super(NodeRNN, self).__init__()
        self.n_sig = n_sig
        self.n_hidden = n_hidden
        self.conv_1 = nn.Conv1d(n_nodes, n_nodes, kernel_size, 1)
        self.conv_2 = nn.Conv1d(n_nodes, n_nodes, kernel_size, 2)
        self.conv_bn = nn.BatchNorm1d(n_nodes)
        self.drop1 = nn.Dropout()
        self.maxpool = nn.MaxPool1d(4, stride=4)
        self.gru = nn.GRU(17, n_hidden, n_layer,
                          bidirectional=True, batch_first=True)
        self.gru_bn = nn.BatchNorm1d(2 * n_hidden)
        self.drop2 = nn.Dropout()
        self.dense = nn.Linear(2 * n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_sig):
        out = self.conv_1(in_sig)
        out = self.conv_2(out)
        out = self.conv_bn(out)
        out = self.drop1(out)
        out = self.maxpool(out)
        out, h = self.gru(out)
        out = torch.cat((out[:, -1, :self.n_hidden], out[:, 0, self.n_hidden:]), 1)
        out = self.gru_bn(out)
        out = self.drop2(out)
        out = self.sigmoid(self.dense(out))
        return out

    def reset_parameters(self):
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.gru.reset_parameters()
        self.dense.reset_parameters()
