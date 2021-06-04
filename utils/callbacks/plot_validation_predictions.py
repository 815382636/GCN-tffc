import torch
import matplotlib.pyplot as plt
from utils.callbacks.base import BestEpochCallback
import os


class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor="", mode="min", data="losloop", model_name="TCN"):
        super(PlotValidationPredictionsCallback, self).__init__(monitor=monitor, mode=mode)
        self.ground_truths = []
        self.predictions = []
        self.data = data
        self.model_name = model_name

    def on_fit_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if trainer.current_epoch != self.best_epoch:
            return
        self.ground_truths.clear()
        self.predictions.clear()
        predictions, y = outputs
        self.ground_truths.append(y[:, 0, :])
        self.predictions.append(predictions[:, 0, :])

    def on_fit_end(self, trainer, pl_module):
        ground_truth = torch.cat(self.ground_truths, dim=0).cpu().numpy()
        predictions = torch.cat(self.predictions, dim=0).cpu().numpy()
        tensorboard = pl_module.logger.experiment
        if not os.path.exists('img'):
            os.system("mkdir -p img")
        if not os.path.exists(f'img/{self.data}'):
            os.system(f"mkdir -p img/{self.data}")
        for node_idx in range(ground_truth.shape[1]):
            plt.clf()
            plt.rcParams["font.serif"] = "Times New Roman"
            fig = plt.figure(figsize=(7, 2), dpi=300)
            plt.plot(
                ground_truth[:, node_idx],
                color="dimgray",
                linestyle="-",
                label="Ground truth",
            )
            plt.plot(
                predictions[:, node_idx],
                color="deepskyblue",
                linestyle="-",
                label="Predictions",
            )
            plt.legend(loc="best", fontsize=10)
            plt.xlabel("Time")
            plt.ylabel("Traffic Speed")
            plt.savefig(f'img/{self.data}/{self.data}_{node_idx}_{self.model_name}.jpg')
            tensorboard.add_figure(
                "Prediction result of node " + str(node_idx),
                fig,
                global_step=len(trainer.train_dataloader) * self.best_epoch,
                close=True,
            )
