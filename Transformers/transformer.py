from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re
from config import *

# =============================================================================
# Transformer
# =============================================================================
def attention(q, k, v, mask = None, dropout = None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])
    
    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)
    
    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

class AttentionHead(nn.Module):
    """
    One head of the self-attention layer
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.head_size = kwargs.get("head_size",4)
        self.num_embed = kwargs.get("num_embed",32)
        self.block_size = kwargs.get("block_size",8)
        self.dropout = kwargs.get("dropout",0.2)

        self.key = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.query = nn.Linear(self.num_embed, self.head_size, bias=False)
        self.value = nn.Linear(self.num_embed, self.head_size, bias=False)
        # tril is a lower triangular matrix. it is not a parameter
        # of the model, so we assign it to the module using register_buffer
        self.register_buffer("tril", torch.tril(torch.ones(self.block_size, self.block_size)))

        # let's also add dropout
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Tril matrix (lower triagular matrix) is used to mask 
        # future positions (setting them to -inf) so that the
        # decoder "learns" to predict next words
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        # weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()

        self.algo = kwargs.get("algo","bert")

        if self.algo == "bert":

            self.n_heads = kwargs.get("n_heads", N_HEADS)
            self.out_dim = kwargs.get("out_dim", EMBED_SIZE)
            self.dropout_bert = kwargs.get("dropout",0.1)

            self.linear = nn.Linear(self.out_dim, self.out_dim*3)

            self.n_heads = self.n_heads
            self.out_dim = self.out_dim
            self.out_dim_per_head = self.out_dim // self.n_heads
            self.out = nn.Linear(self.out_dim, self.out_dim)
            self.dropout = nn.Dropout(self.dropout_bert)

        elif self.algo == "gpt":

            self.num_heads = kwargs.get("num_heads",4)
            self.head_size = kwargs.get("head_size",8)
            self.num_embed = kwargs.get("num_embed",32)
            self.block_size = kwargs.get("block_size",8)
            self.dropout_gpt = kwargs.get("dropout",0.2)

            self.heads = nn.ModuleList(
                [
                    AttentionHead(
                        head_size=self.head_size,
                        num_embed=self.num_embed,
                        block_size=self.block_size,
                        dropout=self.dropout_gpt,
                    )
                    for _ in range(self.num_heads)
                ]
            )
            self.proj = nn.Linear(self.num_embed, self.num_embed)
            self.dropout = nn.Dropout(self.dropout_gpt)
        
        elif self.algo == "vit":

            self.embedding_dim = kwargs.get("embedding_dim",768) 
            self.num_heads = kwargs.get("num_heads",12) 
            self.attn_dropout = kwargs.get("attn_dropout",0)

            # 3. Create the Norm layer (LN)
            self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
            
            # 4. Create the Multi-Head Attention (MSA) layer
            self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim,
                                                        num_heads=self.num_heads,
                                                        dropout=self.attn_dropout,
                                                        batch_first=True) # does our batch dimension come first?


    
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)
    
    def forward(self, x, y=None, mask=None):
        if self.algo == "bert":
            #in decoder, y comes from encoder. In encoder, y=x
            y = x if y is None else y
            
            qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
            q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
            k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
            v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L
            
            #break into n_heads
            q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
            q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
            
            #n_heads => attention => merge the heads => mix information
            scores = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
            scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
            out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE
            
            return out
        
        elif self.algo == "gpt":

            # output of the self-attention
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            # apply the linear projection layer
            out = self.dropout(self.proj(out))
            return out
        
        elif self.algo == "vit":

            x = self.layer_norm(x)
            attn_output, _ = self.multihead_attn(query=x, # query embeddings 
                                                key=x, # key embeddings
                                                value=x, # value embeddings
                                                need_weights=False) # do we need the weights or just the layer outputs?
            return attn_output
        

class FeedForward(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()

        self.algo = kwargs.get("algo","bert")

        if self.algo == "bert":

            self.inp_dim = kwargs.get("inp_dim", EMBED_SIZE)
            self.inner_dim = kwargs.get("inner_dim", INNER_FF_SIZE)
            self.dropout_bert = kwargs.get("dropout", 0.1)

            self.linear1 = nn.Linear(self.inp_dim, self.inner_dim)
            self.linear2 = nn.Linear(self.inner_dim, self.inp_dim)
            self.dropout = nn.Dropout(self.dropout_bert)

        elif self.algo == "gpt":

            self.num_embed = kwargs.get("num_embed",32)
            self.dropout_gpt = kwargs.get("dropout",0.2)

            self.net = nn.Sequential(
                # in the Attention is All You Need paper
                # authors are using the size of the ffwd layer 2048
                # and the output of the model is 512
                # so we apply the same factor of 4
                nn.Linear(self.num_embed, 4 * self.num_embed),
                nn.ReLU(),
                # apply the linear projection layer
                nn.Linear(4 * self.num_embed, self.num_embed),
                nn.Dropout(self.dropout_gpt),
            )

    
    def forward(self, x):
        if self.algo == "bert":
            #inp => inner => relu => dropout => inner => inp
            return self.linear2(self.dropout(F.relu(self.linear1(x)))) 
        elif self.algo == "gpt":
            return self.net(x)

class EncoderLayer(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()

        self.n_heads = kwargs.get("n_heads",N_HEADS)
        self.inner_transformer_size = kwargs.get("inner_transformer_size",EMBED_SIZE)
        self.inner_ff_size = kwargs.get("inner_ff_size",INNER_FF_SIZE)
        self.dropout_gpt = kwargs.get("dropout",0.1)

        self.algo = kwargs.get("algo","bert")

        self.mha = MultiHeadAttention(
            n_heads = self.n_heads, 
            out_dim = self.inner_transformer_size, 
            dropout = self.dropout_gpt,
            algo = self.algo
        )

        self.ff = FeedForward(
            inp_dim = self.inner_transformer_size, 
            inner_dim = self.inner_ff_size, 
            dropout = self.dropout_gpt,
            algo = self.algo
        )
        
        self.norm1 = nn.LayerNorm(self.inner_transformer_size)
        self.norm2 = nn.LayerNorm(self.inner_transformer_size)
        self.dropout1 = nn.Dropout(self.dropout_gpt)
        self.dropout2 = nn.Dropout(self.dropout_gpt)
    
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class TransformerBlock(nn.Module):
    """
    This calss will group together MultiHead Attention and
    FeedForward NN, so that we can copy it in Transformer
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.algo = kwargs.get("algo","gpt")

        self.num_heads = kwargs.get("num_heads", 4)
        self.block_size = kwargs.get("block_size", 8)
        self.num_embed = kwargs.get("num_embed", 32)
        self.dropout_gpt = kwargs.get("dropout", 0.2)

        head_size = self.num_embed // self.num_heads
        self.sa = MultiHeadAttention(
            num_heads=self.num_heads,
            head_size=head_size,
            num_embed=self.num_embed,
            block_size=self.block_size,
            dropout=self.dropout_gpt,
            algo = "gpt"
        )
        self.ffwd = FeedForward(num_embed=self.num_embed, dropout=self.dropout_gpt,algo = "gpt")
        # add the layer normalization
        self.ln1 = nn.LayerNorm(self.num_embed)
        self.ln2 = nn.LayerNorm(self.num_embed)

    def forward(self, x):
        # "x +" is the skip (or residual) connection
        # it helps with optimization
        # also we apply layer normalization before self-attention
        # and feed-forward (a reshufle from original paper)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()
        
        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        # 4. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer.."
        )
    
    # 5. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

# 1. Create a class that inherits from nn.Module
class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0): # Amount of dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        self.msa_block = MultiHeadAttention(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout,
                                                     algo = "vit")
        
        # 4. Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)
        
    # 5. Create a forward() method  
    def forward(self, x):
        
        # 6. Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x 
        
        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x 
        
        return x
    
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """ 
    # 2. Initialize the class with appropriate variables
    def __init__(self, 
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        self.patch_size = patch_size
        
        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method 
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        
        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched) 
        # 6. Make sure the output shape has the right order 
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


class Transformer(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()

        self.algo = kwargs.get("algo","bert")

        if self.algo == "bert":

            self.n_code = kwargs.get("n_code",N_CODE)
            self.n_heads = kwargs.get("n_heads",N_HEADS)
            self.embed_size = kwargs.get("embed_size",EMBED_SIZE)
            self.inner_ff_size = kwargs.get("inner_ff_size",INNER_FF_SIZE)
            self.n_embeddings = kwargs.get("n_embeddings",23948)
            self.seq_len = kwargs.get("seq_len",SEQ_LEN)
            self.dropout = kwargs.get("dropout",0.1)
            
            #model input
            self.embeddings = nn.Embedding(self.n_embeddings, self.embed_size)
            self.pe = PositionalEmbedding(
                d_model = self.embed_size, 
                max_seq_len = self.seq_len
            )
            
            #backbone
            encoders = []
            for i in range(self.n_code):
                encoders += [EncoderLayer(
                    n_heads = self.n_heads, 
                    inner_transformer_size = self.embed_size, 
                    inner_ff_size = self.inner_ff_size, 
                    dropout = self.dropout,
                    algo = self.algo
                    )
                ]

            self.encoders = nn.ModuleList(encoders)
            
            #language model
            self.norm = nn.LayerNorm(self.embed_size)
            self.linear = nn.Linear(self.embed_size, self.n_embeddings, bias=False)

        elif self.algo == "gpt":

            self.vocab_size = kwargs.get("vocab_size", 100)
            self.num_embed = kwargs.get("num_embed", 32)
            self.block_size = kwargs.get("block_size", 8)
            self.num_heads = kwargs.get("num_heads", 4)
            self.num_layers = kwargs.get("num_layers", 4)
            self.dropout_gpt = kwargs.get("dropout", 0.2)
            # each token reads the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(self.vocab_size, self.num_embed)
            # each position from 0 to block_size-1 will get its embedding
            self.position_embedding_table = nn.Embedding(self.block_size, self.num_embed)
            self.blocks = nn.Sequential(
                *[
                    TransformerBlock(
                        num_heads=self.num_heads,
                        block_size=self.block_size,
                        num_embed=self.num_embed,
                        dropout=self.dropout_gpt,
                        algo = self.algo
                    )
                    for _ in range(self.num_layers)
                ]
            )
            # we add the layer norm before the Linear layer
            self.ln_f = nn.LayerNorm(self.num_embed)
            self.lm_head = nn.Linear(self.num_embed, self.vocab_size)

        elif self.algo == "vit":

            self.img_size = kwargs.get("img_size",224)
            self.in_channels = kwargs.get("in_channels",3)
            self.patch_size = kwargs.get("patch_size",16)
            self.num_transformer_layers = kwargs.get("num_transformer_layers",12)
            self.embedding_dim = kwargs.get("embedding_dim",768)
            self.mlp_size = kwargs.get("mlp_size",3072)
            self.num_heads = kwargs.get("num_heads",12)
            self.attn_dropout = kwargs.get("attn_dropout",0)
            self.mlp_dropout = kwargs.get("mlp_dropout",0.1)
            self.embedding_dropout = kwargs.get("embedding_dropout",0.1)
            self.num_classes = kwargs.get("num_classes",1000)

            assert self.img_size % self.patch_size == 0, f"Image size must be divisible by patch size, image size: {self.img_size}, patch size: {self.patch_size}."
        
            # 4. Calculate number of patches (height * width/patch^2)
            self.num_patches = (self.img_size * self.img_size) // self.patch_size**2
                    
            # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
            self.class_embedding = nn.Parameter(data=torch.randn(1, 1, self.embedding_dim),
                                                requires_grad=True)
            
            # 6. Create learnable position embedding
            self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, self.embedding_dim),
                                                requires_grad=True)
                    
            # 7. Create embedding dropout value
            self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)
            
            # 8. Create patch embedding layer
            self.patch_embedding = PatchEmbedding(in_channels=self.in_channels,
                                                patch_size=self.patch_size,
                                                embedding_dim=self.embedding_dim)
            
            # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
            # Note: The "*" means "all"
            self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=self.embedding_dim,
                                                                                num_heads=self.num_heads,
                                                                                mlp_size=self.mlp_size,
                                                                                mlp_dropout=self.mlp_dropout) for _ in range(self.num_transformer_layers)])
        
            # 10. Create classifier head
            self.classifier = nn.Sequential(
                nn.LayerNorm(normalized_shape=self.embedding_dim),
                nn.Linear(in_features=self.embedding_dim, 
                        out_features=self.num_classes)
            )
                
    
    def forward(self, x, targets = None):
        if self.algo == "bert":
            x = self.embeddings(x)
            x = x + self.pe(x)
            for encoder in self.encoders:
                x = encoder(x)
            x = self.norm(x)
            x = self.linear(x)
            return x
        
        elif self.algo == "gpt":

            B, T = x.shape
            # idx --> x and targets are (B,T) tensor of integers
            # the token_emb is (B, T, C), C = NUM_EMBED
            token_emb = self.token_embedding_table(x)
            # (T, C)
            posit_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))

            x = token_emb + posit_emb
            # apply one head of self-attention
            x = self.blocks(x)
            # (B, T, vocab_size)
            logits = self.lm_head(x)
            # compute the loss
            if targets != None:
                # cross_entropy accepts inputs in a (batch_size, num_classes)
                # so we need to reformat our logits dimensions to
                # (batch_size * time, dim_vocabulary), time = block_size
                B, T, C = logits.shape
                logits = torch.reshape(logits, (B * T, C))
                targets = torch.reshape(targets, (B * T,))
                loss = F.cross_entropy(logits, targets)
            else:
                loss = None
            return logits, loss

        elif self.algo == "vit":

            # 12. Get batch size
            batch_size = x.shape[0]
            
            # 13. Create class token embedding and expand it to match the batch size (equation 1)
            class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

            # 14. Create patch embedding (equation 1)
            x = self.patch_embedding(x)

            # 15. Concat class embedding and patch embedding (equation 1)
            x = torch.cat((class_token, x), dim=1)

            # 16. Add position embedding to patch embedding (equation 1) 
            x = self.position_embedding + x

            # 17. Run embedding dropout (Appendix B.1)
            x = self.embedding_dropout(x)

            # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
            x = self.transformer_encoder(x)

            # 19. Put 0 index logit through classifier (equation 4)
            x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

            return x
        

    def generate(self, idx: torch.Tensor, max_new_tokens: int, block_size: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Positional Embedding
class PositionalEmbedding(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()

        self.d_model = kwargs.get("d_model")
        self.max_seq_len = kwargs.get("max_seq_len",80)

        pe = torch.zeros(self.max_seq_len, self.d_model)
        pe.requires_grad = False
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:,:x.size(1)] #x.size(1) = seq_len