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
        self.dropout = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, in_sig):
        out = self.dropout(in_sig)
        out = self.conv1d(out)
        out_mu, h_mu = self.gru_mu(out)
        out_mu = torch.cat((out_mu[:, -1, :self.n_hidden], out_mu[:, 0, self.n_hidden:]), 1)

        out_sigma, h_sigma = self.gru_sigma(out)
        out_sigma = torch.cat((out_sigma[:, -1, :self.n_hidden], out_sigma[:, 0, self.n_hidden:]), 1)

        out = self.reparameterize(out_mu, out_sigma)
        return out, out_mu, out_sigma

    def reset_parameters(self):
        self.conv1d.reset_parameters()
        self.gru.reset_parameters()
        self.linear.reset_parameters()


class DecoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, dropout):
        super(DecoderRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.dropout = dropout
        self.gru = nn.GRU(input_dim, hidden_dim, n_layer, bidirectional=False, batch_first=True)

    def forward(self, input, hidden):
        out, hidden = self.gru(input, hidden)
        return out, hidden


class SigAutoEncoder(nn.Module):
    def __init__(self, input_dim, conv1_layers, kernel_size, hidden_enc, hidden_dec, dropout):
        super(SigAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoder = EncoderRNN(input_dim, conv1_layers, hidden_enc, n_layer=1, kernel_size=kernel_size, dropout=dropout)
        self.decoder = DecoderRNN(input_dim, hidden_dec, n_layer=1, dropout=dropout)
        self.linear = nn.Linear(hidden_dec, input_dim)


    def forward(self, input):
        batch_size = input.size(0)
        seq_len = input.size(2)
        z, mu, logvar = self.encoder(input)
        input = input.squeeze()
        reconstructed = torch.zeros(batch_size, seq_len, self.input_dim).cuda()

        for i in range(0, seq_len):
            if i == 0:
                inp = input[:, i+1:]
                # inp = input[:, 1:].unsqueeze(1)
                _, hidden = self.decoder(inp, z.unsqueeze(0))
            else:
                inp = torch.cat((input[:, 0:i], input[:, i+1:]), 1)
                # inp = input[:, i+1:].unsqueeze(1)
                _, hidden = self.decoder(inp, hidden)
            output = self.linear(hidden.squeeze())
            reconstructed[:, i] = output
        return reconstructed, mu, logvar
        # input = torch.zeros()
        # for i in range(1, self.seq_len):
        #     out[i]
