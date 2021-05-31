import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class GCN(nn.Module):
    def __init__(self, adj, input_dim: int, output_dim: int, **kwargs):
        super(GCN, self).__init__()
        self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim  # seq_len for prediction
        self._output_dim = output_dim  # hidden_dim for prediction
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('tanh'))

    def forward(self, inputs):
        # (batch_size, num_nodes, seq_len)
        batch_size = inputs.shape[0]
        # (num_nodes, batch_size, seq_len)
        inputs = inputs.transpose(0, 1)
        # (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
        # AX (num_nodes, batch_size * seq_len)
        ax = self.laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs

    @property
    def hyperparameters(self):
        return {
            'input_dim': self._input_dim,
            'output_dim': self._output_dim
        }


class MSTTGCN(nn.Module):
    def __init__(self, adj, num_inputs, num_channels, val_num=1, kernel_size=2, dropout=0.2):
        """
            adj: 邻接矩阵
            val_num: 特征数量
            num_inputs: 图卷积后特征的表示维度
            num_channels: 每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
            kernel_size: int, 卷积核尺寸
            dropout: float, drop_out比率
        """
        super(MSTTGCN, self).__init__()
        self._node_num = adj.shape[0]
        self._hidden_dim = num_channels[-1]
        self.register_buffer('adj', torch.FloatTensor(adj))
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.gcn = GCN(adj, val_num, num_inputs)
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._node_num == num_nodes
        gcn_output = []
        for i in range(seq_len):
            output = self.gcn(inputs[:, i, :].reshape((batch_size, num_nodes, 1)))
            gcn_output.append(output)
        gcn_output = torch.cat(gcn_output, 0)
        end_output = []
        for i in range(num_nodes):
            output = self.network(gcn_output[:, :, i, :])
            end_output.append(output)
        end_output = torch.cat(end_output, 0)
        end_output = end_output.transpose(0, 1)
        return end_output[:, :, :, -1]

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
