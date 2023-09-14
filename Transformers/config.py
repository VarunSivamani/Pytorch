import os
import torch
from torch import nn
from torchvision import transforms

# BERT
BATCH_SIZE = 1024
SEQ_LEN = 20
EMBED_SIZE = 128
INNER_FF_SIZE = EMBED_SIZE * 4
N_HEADS = 8
N_CODE = 8
N_VOCAB = 40000
DROPOUT = 0.1
DEVICE = "cuda"

OPTIM_KWARGS = {'lr':1e-4, 'weight_decay':1e-4, 'betas':(.9,.999)}

PTH = './data/'

# GPT
MAX_ITER = 500
# hyperparameters
BATCH_SIZE = 32  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 64  # what is the maximum context length for predictions?
MAX_ITER = 5000  # number of training iterations
EVAL_INTER = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_HEAD = 6
NUM_EMBED = NUM_HEAD * 128
NUM_LAYER = 6
DROPOUT = 0.2

# ViT
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024
IMAGE_PATH = "./pizza_steak_sushi"
TRAIN_DIR = IMAGE_PATH + "/train"
TEST_DIR = IMAGE_PATH + "/test"
NUM_WORKERS = os.cpu_count()-1
IMG_SIZE = 224

MANUAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])  