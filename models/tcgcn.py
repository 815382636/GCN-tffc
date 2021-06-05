import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
from models.tcn import Chomp1d
from models.gcn import GCN
import argparse


class TB(nn.Module):
    def __init__(self, adj, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TB, self).__init__()
        self._outputs = n_outputs
        self.register_buffer('adj', torch.FloatTensor(adj))
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.LeakyReLU1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.LeakyReLU2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.LeakyReLU1, self.dropout1,
                                 self.conv2, self.chomp2, self.LeakyReLU2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.LeakyReLU = nn.LeakyReLU()
        self.init_weights()
        self.gcn = GCN(adj, n_outputs, n_outputs)

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        batch_size, num_nodes, val_num, seq_len = x.shape
        x = x.reshape(batch_size * num_nodes, val_num, seq_len)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        output = self.LeakyReLU(out + res)
        output = output.reshape(batch_size, num_nodes, self._outputs, seq_len)
        output = output.transpose(1, 3)
        output = output.reshape(batch_size * seq_len, self._outputs, num_nodes)
        output = self.gcn(output)
        output = output.reshape(batch_size, seq_len, num_nodes, self._outputs)
        output = output.transpose(1, 2).transpose(2, 3)
        return output



class TCGCN(nn.Module):
    def __init__(self, adj, num_channels, val_num=1, kernel_size=2, dropout=0.2):
        super(TCGCN, self).__init__()
        self._node_num = adj.shape[0]
        self._hidden_dim = num_channels[-1]
        self.register_buffer('adj', torch.FloatTensor(adj))
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = val_num if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TB(adj, in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                          padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._node_num == num_nodes
        inputs = inputs.transpose(1, 2)
        inputs = inputs.reshape(batch_size, num_nodes, 1, seq_len)
        output = self.network(inputs)[:, :, :, -1]
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._node_num,
            'hidden_dim': self._hidden_dim
        }
