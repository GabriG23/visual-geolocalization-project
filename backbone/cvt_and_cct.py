import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange


def cvt_initialization(fc_output_dim):
    return CompactTransformer(224, 16, dim=fc_output_dim, depth=12, heads=12, dim_head=64, scale_dim=4)           # fc_output_dim era il numero delle classi
 
def cct_initialization(fc_output_dim):
    return CompactTransformer(224, 8, dim=fc_output_dim, conv_embed=True, depth=4, heads=4, dim_head=32, scale_dim=2) # dim = 768

class CompactTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, dim_head, scale_dim, pool='cls', in_channels=3, dropout=0.1, emb_dropout=0.1, conv_embed=False):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2                       # number of patches 224 // 16 (floor division) = 14
        patch_dim = in_channels * patch_size ** 2                           # dimesnion of patch -> 3 * 16 ^ 2 = 2304

        if conv_embed:      # convolutional embedding, only for cct
            self.to_patch_embedding = ConvEmbed(in_channels, dim)
            num_patches = self.to_patch_embedding.sequence_length()
        else:               # sequential embedding, only for cvt
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.pool = nn.Linear(dim, 1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            #nn.Linear(dim, dim)
        )
        self.apply(self.init_weight)

    def forward(self, img):
        x = self.to_patch_embedding(img)        # apply convolutional or sequential embedding
        b, n, _ = x.shape

        x += self.pos_embedding[:, :(n + 1)]    # nn.Parameter(torch.randn(1, num_patches, dim))
        x = self.dropout(x)                     # nn.Dropout(emb_dropout)

        x = self.transformer(x)                 # Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        g = self.pool(x)                        # Sequence Pooling nn.Linear(dim, 1)
        xl = F.softmax(g, dim=1)
        x = einsum('b n l, b n d -> b l d', xl, x)

        # se non squeezo, cosa succede???
        # esco con [batch_size, numm_classes, ]
       
        # arriva con [32, 1, 224]
        # con lo squeeze diventa [32, 224]
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

##### Feed Forward #####
class FeedForward(nn.Module):                           # This class defines a feed-forward neural network module with GELU activation and dropout.
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),                 # another linear layer that transforn
            nn.Dropout(dropout)                         # randomly sets elements of the tensor to zero, helping to prevent overfitting
        )

    def forward(self, x):
        return self.net(x)

##### Attention Module #####
class Attention(nn.Module):                             # implements the self-attention mechanism used in transformers.
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads                                # 64 * 8 = 512
        project_out = not (heads == 1 and dim_head == dim)           # 

        self.heads = heads                                           # 8
        self.scale = dim_head ** -0.20                                # 64^(-0.5) = 0.125

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)    # apply linear module

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = torch.softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

##### Residual Block #####
class Residual(nn.Module):                             # adds the output of a given function to its input
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
##### Pre Normalization #####
class PreNorm(nn.Module):                              # applies layer normalization before passing the input to a given function
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    


##### CONVOLUTIONAL EMBEDDING #####
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


    # def forward(self, x):
    #     b, n, _, h = *x.shape, self.heads
    #     qkv = self.to_qkv(x).chunk(3, dim = -1)
    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)   # B N (H D) -> B H N D'

    #     dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale             # B H I D, B H J D -> B H I J

    #     attn = dots.softmax(dim=-1)

    #     out = einsum('b h i j, b h j d -> b h i d', attn, v)                        # B H I J, B H J D -> B H I D
    #     out = rearrange(out, 'b h n d -> b n (h d)')                                # B H N D -> B N (H D)
    #     out =  self.to_out(out)
    #     return out
