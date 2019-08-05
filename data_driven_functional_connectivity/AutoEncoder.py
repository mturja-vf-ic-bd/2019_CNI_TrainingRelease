import torch
import torch.nn as nn
from data_driven_functional_connectivity.model import NodeConv


class EncoderRNN(nn.Module):
    def __init__(self, feat_dim, conv1d_layers, n_hidden, n_layer,
                 kernel_size, dropout):
        super(EncoderRNN, self).__init__()
        for i in range(len(conv1d_layers) - 1):
            feat_dim = (feat_dim - kernel_size + 1) // kernel_size
        feat_dim = feat_dim - kernel_size + 1
        self.n_hidden = n_hidden
        self.conv1d = NodeConv(conv1d_layers, kernel_size, dropout)
        self.gru_mu = nn.GRU(feat_dim, n_hidden, n_layer,
                          bidirectional=True, batch_first=True)
        self.gru_sigma = nn.GRU(feat_dim, n_hidden, n_layer,
                             bidirectional=True, batch_first=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, in_sig):
        out = self.conv1d(in_sig)

        out_mu, h_mu = self.gru_mu(out)
        out_mu = torch.cat((out_mu[:, -1, :self.n_hidden], out_mu[:, 0, self.n_hidden:]), 1)

        out_sigma, h_sigma = self.gru_sigma(out)
        out_sigma = torch.cat((out_sigma[:, -1, :self.n_hidden], out_sigma[:, 0, self.n_hidden:]), 1)

        out = self.reparameterize(out_mu, out_sigma)
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


class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()