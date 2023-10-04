import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import *


class ClipDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
        

    def setup(self, stage="None"):
        self.train_df , self.val_df = make_train_valid_dfs()
        self.train_loader = build_loaders(self.train_df, self.tokenizer, mode="train")
        self.val_loader = build_loaders(self.val_df, self.tokenizer, mode="valid")

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader