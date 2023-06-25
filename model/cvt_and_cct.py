import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange


def cvt_initialization(fc_output_dim): # CVT
    return CompactTransformer(224, 16, dim=fc_output_dim, depth=12, heads=12) 
 
def cct_initialization(fc_output_dim): # CCT
    return CompactTransformer(224, 16, dim=fc_output_dim, conv_embed=True, depth=6, heads=4)

class CompactTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, pool='cls', dim_head=64, scale_dim=4, in_channels=3, dropout=0.1, emb_dropout=0.1, conv_embed=False):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # qui usiamo solo cls, non c'è GeM anche se si potrebbe mettere

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2                       # number of patches 224 // 16 (floor division) = 14
        patch_dim = in_channels * patch_size ** 2                           # dimension of patch -> 3 * 16 ^ 2 = 768

        if conv_embed:      # convolutional embedding, only for cct
            self.to_patch_embedding = ConvEmbed(in_channels, dim)
            num_patches = self.to_patch_embedding.sequence_length()
        else:               # sequential embedding, only for cvt. b batch, c input channels, h height, w weight, p1 p2 patch size
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),  # convert the image in a sequence of non-overlapping patches
                nn.Linear(patch_dim, dim),  # LINEAR PROJECTION - helps to transform the patch embeddings into a desidred dimension
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout) # transformer

        self.pool = nn.Linear(dim, 1)       # seq pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),              # normalization
            #nn.Linear(dim, num_classes)    # took out linear layer, cosface does this
        )
        self.apply(self.init_weight)

    def forward(self, img):
        x = self.to_patch_embedding(img)        # apply convolutional or sequential embedding, 1st step Embed to patches
        b, n, _ = x.shape # n is the number of patches x = [batch, patch, embedding]

        x += self.pos_embedding[:, :(n + 1)]    # learnable parameter of shape [1, num_patches, dim]. each patch embedding in x is addes to its corresponding position embedding
                                                # aiuta ad incorporare le informazioni spaziali nel patch embeddings, permettendo al modello di catturare relazioni posizionali tra patches 
        x = self.dropout(x)                     # aiuta a regolarizzare le patch embeddings prima di mandarle al transformers (previene overfitting)

        x = self.transformer(x)                 # 

        g = self.pool(x)                        # Sequence Pooling nn.Linear(dim, 1)
        xl = F.softmax(g, dim=1)
        x = einsum('b n l, b n d -> b l d', xl, x)
       
        # arriva con [32, 1, 224]
        # con lo squeeze diventa [32, 224] e dopo fa una normalizzazione e un linear [32, 224]
        return self.mlp_head(x.squeeze(-2))     # take of the last two dimensions

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

##### TRANSFORMER  #####
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1): # mlp_dim è la dimensione della rete feed-forward
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), # MLHA
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))                           # MLP
            ]))

    def forward(self, x):
        for attn, ff in self.layers: # applica un layer ad x, ogni trasformazione consiste nell'attention (MHSA) e un feedforward (MLP)
            x = attn(x) + x
            x = ff(x) + x
        return x

##### Feed Forward #####
class FeedForward(nn.Module):                    
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),     # linear
            nn.GELU(),                      # applica GeLU 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),     # another linear layer that transforn
            nn.Dropout(dropout)             # dropout
        )

    def forward(self, x): # introduce non-linearity and increse the model capacity to capture complex patterns
        return self.net(x)

##### Attention Module #####
class Attention(nn.Module):                             # implements the self-attention mechanism used in transformers.
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads                                # 64 * 8 = 512
        project_out = not (heads == 1 and dim_head == dim)           # ci dice se bisogna fare la proiezione o no

        self.heads = heads                                           
        self.scale = dim_head ** -0.5     # scaling factor per gli attention scores                            

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)    # mappa x alle queries, keys e valori necessari per il self-attention

        self.to_out = nn.Sequential( # riporta l'output alla sua dimensione originale se proj_out è true
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads       # prende le dimensioni
        qkv = self.to_qkv(x) # reshape di x con [batch_size, heads, sequence_length, dim_head]
        q, k, v = qkv.chunk(3, dim=-1)  # il tensore è diviso in queries, keys e values usando chunk sull'ultima dimensione
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)   #q, k e v vengono rearranged per avere dimensione [batch_size, heads, sequence_length, dim_head]
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale   # applica il dot product tra queries e keys

        attn = torch.softmax(dots, dim=-1)  # applica softmax sull'ultima dimensione

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # pesa i valori dall'attention score
        out = rearrange(out, 'b h n d -> b n (h d)') # rearreange per avere dimensione batch_size, sequence_length, inner_dim
        out = self.to_out(out)  # applica proiezione se proj_out = true
        return out

##### Residual Block #####
class Residual(nn.Module):  # enables residuala connection in NN, allowing information from the input to flow through the block unchanged while still incorporating the transformations
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
##### Pre Normalization #####
class PreNorm(nn.Module):       
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):             # normalizza il tensore di input prima di applicare una funzione, aiuta a stabilizzare il training
        return self.fn(self.norm(x), **kwargs)
    
##### CONVOLUTIONAL EMBEDDING MODULE #####
class ConvEmbed(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=7, stride=2, padding=3, pool_kernel_size=3, pool_stride=2, pool_padding=1):
        super(ConvEmbed, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding),
            Rearrange('b d h w -> b (h w) d')               # B D H W -> B (H W) D
            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.conv_layers(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)