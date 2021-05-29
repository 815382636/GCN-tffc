import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import utils
import models
import tasks

DATA_PATHS = {
    'wenyi': {
        'feat': 'data/wenyi_speed.csv',
        'adj': 'data/wentyi_adj.csv'
    }
}


def main_supervised(args):
    # print(DATA_PATHS[args.data]['feat'], DATA_PATHS[args.data]['adj'])
    # print('-' * 20)
    dm = utils.data.SpatioTemporalCSVDataModule(feat_path=DATA_PATHS[args.data]['feat'],
                                                adj_path=DATA_PATHS[args.data]['adj'],
                                                **vars(args))
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=get_callbacks(args))
    trainer.fit(task, dm)


def main(args):
    # print(vars(args))
    rank_zero_info(vars(args))
    globals()['main_' + args.settings](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--data', type=str, help='The name of the dataset',
                        choices=('shenzhen', 'losloop', 'wenyi'), default='wenyi')
    parser.add_argument('--model_name', type=str, help='The name of the model for spatiotemporal prediction',
                        choices=('GCN', 'GRU', 'TGCN', 'STGCN', 'MSTTGCN'), default='MSTTGCN')
    parser.add_argument('--settings', type=str, help='The type of settings, e.g. supervised learning',
                        choices=('supervised',), default='supervised')

    temp_args, _ = parser.parse_known_args()

    # DataModule parser 暂时没影响
    parser = getattr(utils.data, temp_args.settings.capitalize() + 'DataModule').add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + 'ForecastTask').add_task_specific_arguments(parser)

    args = parser.parse_args()
    utils.misc.format_logger(pl._logger)

    main(args)
