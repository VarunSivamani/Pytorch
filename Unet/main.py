import os
import pytorch_lightning as pl
from torchsummary import summary

from Lightning.UNETLit import UNETLitModel
from Lightning.datamodule import OxfordIIITPetDataModule
from model import UNet
from utils import *
from config import *

dm = OxfordIIITPetDataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

dm.setup()
train_loader = dm.train_dataloader()
test_loader = dm.test_dataloader()

def runner(model):

    trainer = pl.Trainer(
        precision="16-mixed",
        max_epochs=NUM_EPOCHS,
        accelerator="gpu"
    )

    trainer.fit(model, dm)