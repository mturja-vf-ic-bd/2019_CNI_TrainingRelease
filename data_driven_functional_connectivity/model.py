import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


def init_param(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()


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
    def __init__(self, channel_seq=[116, 20, 10],
                 kernel_size=5, dropout=0.5, sigmoid=False):
        super(NodeConv, self).__init__()
        list_of_layer = []
        for i in range(0, len(channel_seq)):
            if i == 0:
                list_of_layer.append(
                    nn.Sequential(
                        nn.Conv2d(1, channel_seq[i], (1, kernel_size), 1),
                        nn.MaxPool2d((1, kernel_size), stride=(1, kernel_size)),
                        nn.LeakyReLU(0.1),
                        nn.Dropout(dropout),
                        nn.BatchNorm2d(channel_seq[i])
                    )
                )
            elif i < len(channel_seq) - 1:
                list_of_layer.append(
                    nn.Sequential(
                        nn.Conv2d(channel_seq[i - 1], channel_seq[i], (1, kernel_size), 1),
                        nn.MaxPool2d((1, kernel_size), stride=(1, kernel_size)),
                        nn.LeakyReLU(0.1),
                        nn.Dropout(dropout),
                        nn.BatchNorm2d(channel_seq[i])
                    )
                )
            else:
                list_of_layer.append(nn.Conv2d(channel_seq[i - 1], channel_seq[i], (1, kernel_size), 1))
        self.kernel_size = kernel_size
        self.conv1d = nn.Sequential(*list_of_layer)
        self.sigmoid = nn.Sigmoid()
        self.sigm = sigmoid

    def forward(self, in_sig):
        for i, layer in enumerate(self.conv1d):
            if i == 0:
                out = layer(in_sig)
            else:
                out = layer(out)
        if self.sigm:
            out = self.sigmoid(out.view(-1, 1))
        return out.squeeze()

    def reset_parameters(self):
        for layer in self.conv1d:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                layer.reset_parameters()


class NodeDeConv(nn.Module):
    def __init__(self, channel_seq, kernel_size, stride, dropout):
        super(NodeDeConv, self).__init__()
        list_of_layer = []
        for i in range(1, len(channel_seq)):
            if i < len(channel_seq) - 1:
                list_of_layer.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(channel_seq[i - 1], channel_seq[i], kernel_size=(1, kernel_size), stride=stride[i-1], output_padding=1),
                        nn.LeakyReLU(0.1),
                        nn.Dropout(dropout),
                        nn.BatchNorm2d(channel_seq[i])
                    )
                )
            else:
                list_of_layer.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(channel_seq[i - 1], channel_seq[i], kernel_size=(1, kernel_size), stride=stride[i-1], output_padding=1)
                    )
                )
        self.deconv1d = nn.Sequential(*list_of_layer)

    def forward(self, input):
        return self.deconv1d(input)

    def reset_parameters(self):
        for layer in self.conv1d:
            if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.BatchNorm2d):
                layer.reset_parameters()


class SAE(nn.Module):
    def __init__(self, conv_seq, conv_kernel, deconv_seq, deconv_kernel, deconv_stride, dropout):
        super(SAE, self).__init__()
        self.conv = NodeConv(conv_seq, conv_kernel, dropout)
        self.deconv = NodeDeConv(deconv_seq, deconv_kernel, deconv_stride, dropout)

    def forward(self, input):
        out = self.conv(input)
        out = self.deconv(out)
        return out


class NodeRNN(nn.Module):
    def __init__(self, feat_dim, conv1d_layers, n_hidden, n_layer,
                 kernel_size, dropout):
        super(NodeRNN, self).__init__()
        for i in range(len(conv1d_layers) - 1):
            feat_dim = (feat_dim - kernel_size + 1) // kernel_size
        feat_dim = feat_dim - kernel_size + 1
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.conv1d = NodeConv(conv1d_layers, kernel_size, dropout)
        self.gru = nn.GRU(feat_dim, n_hidden, n_layer,
                          bidirectional=True, batch_first=True)
        self.non_linear = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * n_hidden)
        )
        self.dense = nn.Sequential(nn.Linear(2 * n_hidden, 1) , nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, in_sig):
        out = self.dropout(in_sig)
        out = self.conv1d(out)
        #out = out.view(out.size(0), -1, 1)
        out, h = self.gru(out)
        out = h.view(2, self.n_layer, h.size(1), h.size(2))
        out = torch.cat((out[0, -1], out[1, -1]), 1)
        # out = torch.cat((out[:, -1, :self.n_hidden], out[:, 0, self.n_hidden:]), 1)
        out = self.non_linear(out)
        out = self.dense(out)
        return out

    def reset_parameters(self):
        self.conv1d.reset_parameters()
        self.gru.reset_parameters()
        for layer in self.non_linear:
            if isinstance(layer, nn.BatchNorm1d):
                layer.reset_parameters()
        for layer in self.dense:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
