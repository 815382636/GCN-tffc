# GCN-tffc

Fork frame from [T-GCN-PyTorch](https://img.shields.io/github/stars/martinwhl/T-GCN-PyTorch)

## Requirements

* numpy
* matplotlib
* pandas
* torch
* pytorch-lightning>=1.3.0
* torchmetrics>=0.3.0
* python-dotenv

## Model Training

```bash
# GCN
python main.py --model_name GCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1 --pre_len 1
# GRU
python main.py --model_name GRU --max_epochs 3000 --learning_rate 0.001 --weight_decay 1.5e-3 --batch_size 64 --hidden_dim 100 --settings supervised --gpus 1 --pre_len 1
# T-GCN
python main.py --model_name TGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse_with_regularizer --settings supervised --gpus 1 --pre_len 1
# TCN
python main.py --model_name TCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse --settings supervised --gpus 1 --pre_len 1
# MSTTGCN
python main.py --model_name MSTTGCN --max_epochs 3000 --learning_rate 0.001 --weight_decay 0 --batch_size 32 --hidden_dim 64 --loss mse --settings supervised --gpus 1 --pre_len 1
```

You can also adjust the `--data`, `--seq_len`, `--pre_len`, `--tcn_len`, `--tcn_wid` and  `--email`  parameters.

Run `tensorboard --logdir lightning_logs/version_0` to monitor the training progress and view the prediction results.
