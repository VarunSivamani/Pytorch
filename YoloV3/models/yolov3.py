import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.loss import YoloLoss
import config as cfg

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YoloV3(pl.LightningModule):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        self.criterion = YoloLoss()

        self.metrics = dict(
            train_step = 0,
            val_step = 0,
            train_loss = list(),
            train_acc = list(),
            val_loss = list(),
            val_acc = list(),
            epoch_train_step = 0,
            epoch_val_step = 0,
        )

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers


    def get_layer(self, index):
        if index < len(self.layers) and index >= 0:
            return self.layers[index]


    def training_step(self, train_batch, batch_idx):
        x, target = train_batch
        output = self.forward(x)
        loss = self.criterion(output, target, return_dict=True)
        acc = self.criterion.check_class_accuracy(output, target, cfg.CONF_THRESHOLD)

        self.metrics['train_loss'].append(loss)
        self.metrics['train_acc'].append(acc)
        self.metrics['train_step'] += 1
        self.metrics['epoch_train_step'] += 1

        self.log_dict({'training_loss': loss['total_loss']})

        return loss['total_loss']


    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        output = self.forward(x)
        loss = self.criterion(output, target, return_dict=True)
        acc = self.criterion.check_class_accuracy(output, target, cfg.CONF_THRESHOLD)

        self.metrics['val_loss'].append(loss)
        self.metrics['val_acc'].append(acc)
        self.metrics['val_step'] += 1
        self.metrics['epoch_val_step'] += 1

        self.log_dict({'validation_loss': loss['total_loss']})


    def on_validation_epoch_end(self):
        if self.metrics["train_step"]:
            print("Epoch ", self.current_epoch)
            curr_loss = 0
            curr_acc = dict(
                correct_class = 0,
                correct_noobj = 0,
                correct_obj = 0,
                total_class_preds = 0,
                total_noobj = 0,
                total_obj = 0,
            )

            for i in range(self.metrics["epoch_train_step"]):
                loss_ = self.metrics["train_loss"][i]
                curr_loss += loss_["total_loss"]
                acc = self.metrics["train_acc"][i]
                curr_acc["correct_class"] += acc["correct_class"]
                curr_acc["correct_noobj"] += acc["correct_noobj"]
                curr_acc["correct_obj"] += acc["correct_obj"]
                curr_acc["total_class_preds"] += acc["total_class_preds"]
                curr_acc["total_noobj"] += acc["total_noobj"]
                curr_acc["total_obj"] += acc["total_obj"]

            print("Results of Training :")
            print(f"Class accuracy  : {(curr_acc['correct_class']/(curr_acc['total_class_preds']+1e-16))*100:2f}%")
            print(f"No obj accuracy : {(curr_acc['correct_noobj']/(curr_acc['total_noobj']+1e-16))*100:2f}%")
            print(f"Obj accuracy    : {(curr_acc['correct_obj']/(curr_acc['total_obj']+1e-16))*100:2f}%")
            print(f"Total loss: {(curr_loss/(len(self.metrics['train_loss'])+1e-16)):2f}")

            self.metrics["train_loss"] = []
            self.metrics["train_acc"] = []
            self.metrics["epoch_train_step"] = 0

            curr_loss = 0
            curr_acc = dict(
                correct_class = 0,
                correct_noobj = 0,
                correct_obj = 0,
                total_class_preds = 0,
                total_noobj = 0,
                total_obj = 0,
            )

            for i in range(self.metrics["epoch_val_step"]):
                loss_ = self.metrics["val_loss"][i]
                curr_loss += loss_["total_loss"]
                acc = self.metrics["val_acc"][i]
                curr_acc["correct_class"] += acc["correct_class"]
                curr_acc["correct_noobj"] += acc["correct_noobj"]
                curr_acc["correct_obj"] += acc["correct_obj"]
                curr_acc["total_class_preds"] += acc["total_class_preds"]
                curr_acc["total_noobj"] += acc["total_noobj"]
                curr_acc["total_obj"] += acc["total_obj"]

            print("Results on Validation :")
            print(f"Class accuracy  : {(curr_acc['correct_class']/(curr_acc['total_class_preds']+1e-16))*100:2f}%")
            print(f"No obj accuracy : {(curr_acc['correct_noobj']/(curr_acc['total_noobj']+1e-16))*100:2f}%")
            print(f"Obj accuracy    : {(curr_acc['correct_obj']/(curr_acc['total_obj']+1e-16))*100:2f}%")
            print(f"Total loss: {(curr_loss/(len(self.metrics['val_loss'])+1e-16)):2f}")

            self.metrics["val_loss"] = []
            self.metrics["val_acc"] = []
            self.metrics["epoch_val_step"] = 0

            print("Saving and Creating checkpoint...")
            print("\n")
            self.trainer.save_checkpoint(cfg.CHECKPOINT_FILE)


    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.LEARNING_RATE,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.train_dataloader()),
            pct_start=8 / self.trainer.max_epochs,
            div_factor=100,
            final_div_factor=100,
            three_phase=False,
            verbose=False
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',
                'frequency': 1
            },
        }


def check(model):
    x = torch.randn((2, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, cfg.IMAGE_SIZE // 32, cfg.IMAGE_SIZE // 32, cfg.NUM_CLASSES + 5)
    assert model(x)[1].shape == (2, 3, cfg.IMAGE_SIZE // 16, cfg.IMAGE_SIZE // 16, cfg.NUM_CLASSES + 5)
    assert model(x)[2].shape == (2, 3, cfg.IMAGE_SIZE // 8, cfg.IMAGE_SIZE // 8, cfg.NUM_CLASSES + 5)
    print("Success!")