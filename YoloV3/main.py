from dataset.dataset import *
from models.YoloV3Lightning import *
from utils.utils import *

def main(model, train_loader, val_loader, train_flag=True):

    if train_flag:
        trainer = pl.Trainer(
            precision=16,
            max_epochs=cfg.NUM_EPOCHS,
            accelerator='gpu'
        )

        check_pt = {}
        if cfg.LOAD_MODEL:
            check_pt = dict(ckpt_path=cfg.CHECKPOINT_FILE)

        trainer.fit(model, train_loader, val_loader, **check_pt)

    model.to(cfg.DEVICE)
    model.eval()
    cfg.IMG_DIR = cfg.DATASET + "/images/"
    cfg.LABEL_DIR = cfg.DATASET + "/labels/"

    eval_dataset = YOLODataset(
        cfg.DATASET + "/test.csv",
        transform=cfg.test_transforms,
        S=[cfg.IMAGE_SIZE // 32, cfg.IMAGE_SIZE // 16, cfg.IMAGE_SIZE // 8],
        img_dir=cfg.IMG_DIR,
        label_dir=cfg.LABEL_DIR,
        anchors=cfg.ANCHORS,
        mosaic=False
    )

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    scaled_anchors = (
            torch.tensor(cfg.ANCHORS)
            * torch.tensor(cfg.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    scaled_anchors = scaled_anchors.to(cfg.DEVICE)

    plot_examples(model, eval_loader, 0.5, 0.6, scaled_anchors)

    pred_boxes, true_boxes = get_evaluation_bboxes(
        eval_loader,
        model,
        iou_threshold=cfg.NMS_IOU_THRESH,
        anchors=cfg.ANCHORS,
        threshold=cfg.CONF_THRESHOLD,
    )

    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=cfg.MAP_IOU_THRESH,
        box_format="midpoint",
        num_classes=cfg.NUM_CLASSES,
    )

    print(f"MAP: {mapval.item()}")
