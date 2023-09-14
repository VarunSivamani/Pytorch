from config import *
from utils import *
from transformer import Transformer
from training import bert_training, gpt_training, vit_training
from validation import bert_validation, gpt_validation, vit_validation

def BERT():

    dataset, data_loader = bert_training(PTH, N_VOCAB, SEQ_LEN)

    model = Transformer(
        n_code = 8, 
        n_heads = 8, 
        embed_size = 128, 
        inner_ff_size = 128*4, 
        n_embeddings = len(dataset.vocab), 
        seq_len = 20, 
        dropout = 0.1,
        algo = "bert"
    )
    model = model.cuda()

    bert_validation(model, dataset, data_loader)

    print("End")


def GPT():

    tokenizer, vocab_size, train_data, val_data = gpt_training(PTH)

    model = Transformer(
        vocab_size=vocab_size,
        num_embed=6*128,
        block_size=64,
        num_heads=6,
        num_layers=6,
        dropout=0.2,
        algo = "gpt"
    ).to(DEVICE)

    gpt_validation(model, train_data, val_data, tokenizer)

    print("End")


def ViT():

    train_dataloader, test_dataloader, class_names = vit_training()
    print("Loaded data")
   

    print("Initialising Model")
    # Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
    vit = Transformer(num_classes=len(class_names),algo = "vit")
    print("Model Done")

    vit_validation(vit, train_dataloader, test_dataloader)
    
