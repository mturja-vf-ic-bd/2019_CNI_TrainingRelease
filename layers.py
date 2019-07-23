import torch
import torch.nn as nn


class conv1d_sig(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(conv1d_sig, self).__init__()

