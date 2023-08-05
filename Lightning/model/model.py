import torch
import torchvision
import numpy
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from config import LEARNING_RATE
from torch import nn, optim
import torchmetrics
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch_lr_finder import LRFinder
from utils.metrics import MyAccuracy
import matplotlib.pyplot as plt
from utils.utils import plot_images,plot_graphs,plot_train_images


class ConvLayer(nn.Module):
    def __init__(self, input_c, output_c, dropout, bias=False, stride=1, padding=1, maxpool=False):
        super(ConvLayer, self).__init__()

        block = list()
        block.append(nn.Conv2d(input_c, output_c, 3, bias=bias, stride=stride, padding=padding, padding_mode='replicate'))
        if maxpool:
            block.append(nn.MaxPool2d(2, 2))
        block.append(nn.BatchNorm2d(output_c))
        block.append(nn.ReLU())
        block.append(nn.Dropout(dropout))

        self.blocks = nn.Sequential(*block)

    def forward(self, x):
        return self.blocks(x)


class Custom_Layer(nn.Module):
    def __init__(self, input_c, output_c, dropout, maxpool=True, layers=2):
        super(Custom_Layer, self).__init__()

        self.pool_block = ConvLayer(input_c, output_c, dropout=dropout, maxpool=maxpool)
        self.dropout = dropout
        self.residual_block = None

        if layers > 0:
            layer = list()
            for i in range(0, layers):
                layer.append(ConvLayer(output_c, output_c, dropout=dropout, maxpool=False))

            self.residual_block = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pool_block(x)

        if self.residual_block is not None:
            y = x
            x = self.residual_block(x)
            x = x + y
        return x

class NN(pl.LightningModule):
    def __init__(self, lr=LEARNING_RATE, num_classes=10, max_epochs=24):
        super().__init__()
        self.network = nn.Sequential(
            Custom_Layer(  3,  64, layers=0, dropout=0.01, maxpool=False),
            Custom_Layer( 64, 128, layers=2, dropout=0.01, maxpool=True),
            Custom_Layer(128, 256, layers=0, dropout=0.01, maxpool=True),
            Custom_Layer(256, 512, layers=2, dropout=0.01, maxpool=True),
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = lr
        self.train_accuracy = MyAccuracy()
        self.val_accuracy = MyAccuracy()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.max_epochs = max_epochs
        self.epoch_counter = 1

        self.classes = [
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        self.training_acc = list()
        self.training_loss = list()
        self.testing_acc = list()
        self.testing_loss = list()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.epoch_counter == 1:
            if not hasattr(self, "images_plotted"):
                self.images_plotted = True

                scores = self.forward(x)
                preds = torch.argmax(scores, dim=1)

                self.train_images = []
                self.train_predictions = []
                self.train_labels = []

                for i in range(20):
                    x, target = batch

                    output = self.forward(x)

                    _, preds = torch.max(output, 1)

                    for i in range(len(preds)):
                        if preds[i] != target[i]:
                            self.train_images.append(x[i])
                            self.train_predictions.append(preds[i])
                            self.train_labels.append(target[i])

                plot_train_images(
                    self.train_images,
                    self.train_labels,
                    self.classes,
                )
                print("\n")
            return self._common_step(batch, self.train_loss, self.train_accuracy)

        if batch_idx % 100 == 0:
            x = x[:8]
            grid = torchvision.utils.make_grid(x.view(-1, 3, 32, 32))
            self.logger.experiment.add_image("cifar10_images", grid, self.global_step)

        return self._common_step(batch, self.train_loss, self.train_accuracy)

    def on_train_epoch_end(self):
        self.training_acc.append(self.train_accuracy.compute())
        self.training_loss.append(self.train_loss.compute())
        print(
            f"Epoch: {self.epoch_counter}, Train: Loss: {self.train_loss.compute():0.4f}, Accuracy: "
            f"{self.train_accuracy.compute():0.2f}"
        )
        print("\n")
        self.train_loss.reset()
        self.train_accuracy.reset()
        self.epoch_counter += 1

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, self.val_loss, self.val_accuracy)

        self.log("val_step_loss", self.val_loss, prog_bar=True, logger=True)
        self.log("val_step_acc", self.val_accuracy, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self):
        self.testing_acc.append(self.val_accuracy.compute())
        self.testing_loss.append(self.val_loss.compute())
        print(
            f"Epoch: {self.epoch_counter}, Valid: Loss: {self.val_loss.compute():0.4f}, Accuracy: "
            f"{self.val_accuracy.compute():0.2f}"
        )
        self.val_loss.reset()
        self.val_accuracy.reset()

        if(self.epoch_counter == self.max_epochs):
            if not hasattr(self,"graphs_plotted"):
                self.graphs_plotted = True
                
                train_acc_cpu = [acc.item() for acc in self.training_acc]
                train_loss_cpu = [acc.item() for acc in self.training_loss]
                test_acc_cpu = [acc.item() for acc in self.testing_acc]
                test_loss_cpu = [acc.item() for acc in self.testing_loss]

                plot_graphs(train_loss_cpu, train_acc_cpu, test_loss_cpu, test_acc_cpu)

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, self.val_loss, self.val_accuracy)
        self.log("test_loss", loss)

        return loss

    def _common_step(self, batch, loss_metric, acc_metric):
        x, y = batch
        batch_len = y.numel()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        loss_metric.update(loss, batch_len)
        acc_metric.update(logits, y)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        # x = x.reshape(x.size(0),-1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)

        self.images = []
        self.predictions = []
        self.labels = []

        for i in range(20):
            x, target = batch

            output = self.forward(x)

            _, preds = torch.max(output, 1)

            for i in range(len(preds)):
                if preds[i] != target[i]:
                    self.images.append(x[i])
                    self.predictions.append(preds[i])
                    self.labels.append(target[i])

        return self.images, self.predictions, self.labels

    def train_dataloader(self):

        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def find_lr(self, optimizer):
        lr_finder = LRFinder(self, optimizer, self.criterion)
        lr_finder.range_test(
            self.train_dataloader(), end_lr=0.1, num_iter=100, step_mode="exp"
        )
        _, best_lr = lr_finder.plot()
        lr_finder.reset()
        return best_lr

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-7, weight_decay=1e-2)
        best_lr = self.find_lr(optimizer)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=best_lr,
            epochs=self.max_epochs,
            pct_start=5 / self.max_epochs,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
