import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(self, feat_path: str, adj_path: str, batch_size: int = 64,
                 seq_len: int = 12, pre_len: int = 3,
                 split_ratio: float = 0.8, normalize: bool = True, **kwargs):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = utils.data.functions.load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)
        # print('----------数据模型参数----------')
        # print('self._feat_path：' + self._feat_path)
        # print('self._adj_path：' + self._adj_path)
        # print('self.batch_size：' + str(self.batch_size))
        # print('self.seq_len：' + str(self.seq_len))
        # print('self.pre_len：' + str(self.pre_len))
        # print('self.split_ratio：' + str(self.split_ratio))
        # print('self.normalize：' + str(self.normalize))
        # print('self._feat：')
        # print(self._feat)
        # print('self._feat_max_val：' + str(self._feat_max_val))
        # print('self._adj：')
        # print(self._adj)
        # print('--------------------------')

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        """
            将 数据导入模型参入 加入全局参数
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--seq_len', type=int, default=12)
        parser.add_argument('--pre_len', type=int, default=3)
        parser.add_argument('--split_ratio', type=float, default=0.8)
        parser.add_argument('--normalize', type=bool, default=True)
        return parser

    def setup(self, stage: str = None):
        """
            获取 训练集、测试集 预处理后的 np.array
        """
        self.train_dataset, self.val_dataset = \
            utils.data.functions.generate_torch_datasets(self._feat, self.seq_len, self.pre_len,
                                                         split_ratio=self.split_ratio, normalize=self.normalize)

    def train_dataloader(self):
        """
            pytorch  按照 batch_size 大小 进行 训练集数据输入
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """
            pytorch  按照 batch_size 大小 进行 测试集数据输入
        """
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj
