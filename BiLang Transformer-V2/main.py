import warnings
import pytorch_lightning as pl

from Lightning.datamodule import *
from Lightning.TransformersLitModel import BilangLightning
from config import get_config

warnings.filterwarnings('ignore')
config_ = get_config()

def Runner():
    data = OpusDataModule()
    data.setup()
    src, tgt = data.tokenizer_src, data.tokenizer_tgt

    model = BilangLightning(
        learning_rate=1e-3,
        tokenizer_src=src,
        tokenizer_tgt=tgt
    )

    trainer = pl.Trainer(
        precision="16-mixed",
        max_epochs=config_["num_epochs"],
        accelerator="gpu"
    )

    trainer.fit(model,data)