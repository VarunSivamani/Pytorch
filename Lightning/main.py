import pytorch_lightning as pl
from config import MAX_EPOCHS, device, LEARNING_RATE, BATCH_SIZE, NUM_CLASSES,CLASSES
from utils.metrics import MyPrintingCallback
from model.model import NN
from dataset.dataloader import CIFAR10DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from utils.utils import *
import warnings
from torchsummary import summary

warnings.filterwarnings("ignore")

dm = CIFAR10DataModule(data_dir="data/", batch_size=BATCH_SIZE, num_workers=4)
model = NN(num_classes=NUM_CLASSES, max_epochs=MAX_EPOCHS).to(device)
logger = TensorBoardLogger("tb_logs", name="cifar10_model_v0")

trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=[0],
    min_epochs=1,
    max_epochs=MAX_EPOCHS,
    precision=16,
    callbacks=[MyPrintingCallback()],
)

trainer.fit(model, datamodule=dm)
trainer.validate(model, dm)
trainer.test(model, dm)

if __name__ == '__main__':

    dm = CIFAR10DataModule(data_dir="dataset/", batch_size=BATCH_SIZE, num_workers=4)
    model = NN(num_classes=NUM_CLASSES, max_epochs=MAX_EPOCHS).to(device)
    logger = TensorBoardLogger("tb_logs", name="cifar10_model_v0")

    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[0],
        min_epochs=1,
        max_epochs=MAX_EPOCHS,
        precision=16,
        callbacks=[MyPrintingCallback()],
    )

    trainer.fit(model, datamodule=dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
