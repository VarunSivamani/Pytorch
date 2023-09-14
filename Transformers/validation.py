import numpy as np
import torch.optim as optim
from config import *
from utils import bert_training_loop, gpt_training_loop, decode
from super_repo import data_setup, engine, utils
from super_repo.utils import plot_loss_curves


def bert_validation(model, dataset, data_loader):

    print('Initializing Optimizer and Loss...')
    optimizer = optim.Adam(model.parameters(), **OPTIM_KWARGS)
    loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)

    print('Starting Model Training...')
    print_each = 10
    model.train()
    batch_iter = iter(data_loader)
    n_iteration = 10000

    bert_training_loop(model,loss_model,optimizer,data_loader,batch_iter,print_each,n_iteration)

    print('Saving Embeddings...')
    N = 3000
    np.savetxt('values.tsv', np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
    s = [dataset.rvocab[i] for i in range(N)]
    open('names.tsv', 'w+').write('\n'.join(s) )


def gpt_validation(model, train_data, val_data, tokenizer):

    print(
        "Model with {:.2f}M parameters".format(sum(p.numel() for p in model.parameters()) / 1e6)
    )
    print('Initializing Optimizer ...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    gpt_training_loop(model, optimizer, MAX_ITER, train_data, val_data)

    print("Validation....")

    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print("\n Starting with Validation...")
    print(
        decode(
            enc_sec=model.generate(idx=context, max_new_tokens=100, block_size=BLOCK_SIZE)[0],
            tokenizer=tokenizer,
        )
    )

def vit_validation(vit, train_dataloader, test_dataloader):

    # Create a random tensor with same shape as a single image
    random_image_tensor = torch.randn(1, 3, 224, 224) # (batch_size, color_channels, height, width)

    # Pass the random image tensor to our ViT instance
    vit(random_image_tensor)

    print('Initializing Optimizer and Loss...')
    # Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper 
    optimizer = torch.optim.Adam(params=vit.parameters(), 
                                lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                                betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                                weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k
    
    # Setup the loss function for multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Validation....")
    # # Train the model and save the training results to a dictionary
    results = engine.train(model=vit,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=10,
                        device=DEVICE)
    
    print()
    
    plot_loss_curves(results)