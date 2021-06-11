import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
from torch.nn.utils import weight_norm
from models.gcn import GCN
from models.tcn import TemporalBlock
from torch.nn import Parameter
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, adj, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('adj', torch.FloatTensor(adj))
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input = input.transpose(1, 2)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


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
        self.num_inputs = num_inputs
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.gcn = GraphConvolution(adj, val_num, num_inputs)
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._node_num == num_nodes
        inputs = inputs.reshape(batch_size * seq_len, 1, num_nodes)
        output = self.gcn(inputs)
        output = output.reshape(batch_size, seq_len, num_nodes, self.num_inputs)
        output = output.transpose(1, 2).transpose(2, 3)
        output = output.reshape(batch_size * num_nodes, self.num_inputs, seq_len)
        output = self.network(output)[:, :, -1]
        output = output.reshape(batch_size, num_nodes, self._hidden_dim)
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
