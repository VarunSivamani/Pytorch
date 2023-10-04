from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from config import CFG
import pytorch_lightning as pl
import itertools

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class CLIPModel(pl.LightningModule):

    def __init__(self,temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding) -> None:
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

        self.train_losses = []
        self.val_losses = []

        self.params = [
            {"params": self.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": self.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": itertools.chain(
                self.image_projection.parameters(), self.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]

        self.epoch_condition = 0

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
    def training_step(self, batch, batch_idx):

        loss = self.forward(batch)

        self.log("train_loss",loss,prog_bar=True)
        self.train_losses.append(loss.item())
        
        mean_loss = sum(self.train_losses) / len(self.train_losses)
        self.log("Mean Train Loss", mean_loss, prog_bar=True)

        self.epoch_condition = 1

        return loss

    def validation_step(self, batch, batch_idx):
        
        loss = self.forward(batch)
        self.log("val_loss",loss,prog_bar=True)
        self.val_losses.append(loss.item())

    def on_validation_epoch_end(self):

        if self.epoch_condition > 0:

            print(f"Epoch : {self.current_epoch+1}")

            train_loss_mean = sum(self.train_losses) / len(self.train_losses)
            print(f"Training Loss : {train_loss_mean:0.4f}")

            val_loss_mean = sum(self.val_losses) / len(self.val_losses)
            print(f"Validation Loss : {val_loss_mean:0.4f}")

            self.train_losses = []
            self.val_losses = []

            self.epoch_condition = 0

            if self.current_epoch+1 == self.trainer.max_epochs:
                torch.save(self.state_dict(), "best.pt")
                print("Saving Model")

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.params, weight_decay=0.)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'monitor': 'val_loss'  # Monitor validation loss for reducing LR
            }
        }
        


