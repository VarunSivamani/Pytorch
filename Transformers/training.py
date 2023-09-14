import re
from collections import Counter
from os.path import exists
import torch
from dataset import SentencesDataset
from transformers import AutoTokenizer  # pip install transformers
from utils import encode, create_dataloaders
from config import TRAIN_DIR, TEST_DIR, MANUAL_TRANSFORMS, BATCH_SIZE, NUM_WORKERS

def bert_training(PTH, N_VOCAB, SEQ_LEN):

    print('Loading Text...')
    sentences = open(PTH+'training.txt').read().lower().split('\n')
    
    print('Tokenizing sentences...')
    special_chars = ',?;.:/*!+-()[]{}"\'&'
    sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
    sentences = [[w for w in s if len(w)] for s in sentences]

    print('Creating / Loading Vocab...')
    pth = PTH + 'vocab.txt'
    if not exists(pth):
        words = [w for s in sentences for w in s]
        vocab = Counter(words).most_common(N_VOCAB) #keep the N most frequent words
        vocab = [w[0] for w in vocab]
        open(pth, 'w+').write('\n'.join(vocab))
    else:
        vocab = open(pth).read().split('\n')

    print('Creating Dataset...')
    dataset = SentencesDataset(sentences, vocab, SEQ_LEN)
    # kwargs = {'num_workers':n_workers, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    kwargs = {'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':1024}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    print('Initializing Model...')

    return dataset, data_loader


def gpt_training(PTH):

    print('Loading Text...')
    data_raw = open(PTH + "english.txt", encoding="utf-8").read()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    data = encode(text_seq=data_raw, tokenizer=tokenizer)
    print("Train / Validation Split...")
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    print('Initializing Model...')

    return tokenizer, vocab_size, train_data, val_data

def vit_training():

    print('Loading Text...')
    train_dataloader, test_dataloader, class_names = create_dataloaders(TRAIN_DIR, TEST_DIR, MANUAL_TRANSFORMS, 32,NUM_WORKERS)
    print("Loaded data")

    return train_dataloader, test_dataloader, class_names