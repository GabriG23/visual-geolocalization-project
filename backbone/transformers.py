import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from .stochastic_depth import DropPath

class Attention(Module): # MHSA layer Multi-Headed Self-Attention
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads                          # numero head
        head_dim = dim // self.num_heads                    # divisione intera 
        self.scale = head_dim ** -0.5                       # potenza di - 1/2

        self.qkv = Linear(dim, dim * 3, bias=False)         # 
        self.attn_drop = Dropout(attention_dropout)         # butta fuori l'attention dropout
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)
# B = 32, N = 16385 C = 128, heads = 2, embedding 128
# qkv = Linear(dim, dim*3, bias, False) # 128* 3 = 384
# x input = 32 16385 128
# x = self.qkv(x)                                         -> 32 16385 384
# x.reshape(B, N, 3, self.num_heads, C // self.num_heads) -> 32 16385 3 2 64
# x. permute(2, 0, 3, 1, 4)                               -> 3 32 2 16385 64
# q, k, v = qkv[0], qkv[1], qkv[2]
# q = qkv[0] = 32 2 16385 64
# k = qkv[1] = 32 2 16385 64
# v = qkv[2] ) 32 2 16385 64

# k.transpose(-2, -1) -> 32 2 64 16385
# q @ k prodotto matriciale tra 32 2 64 16385 e 32 2 16385 64 -> 32 2 16385 16385, va in segmentation

    def forward(self, x):
        B, N, C = x.shape           # B N C = 32 16385 128
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # fault segmentation qui
        print(q.shape)
        print(k.shape)
        print(v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale     # non riesce a fare questo prodotto matriciale, la dim di 512 è troppo grande, per questo metteva 224
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """
                       # 128      2
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu                                                        

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)                                                                                       # Layer Normalization
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))                                      # GeLU activation
        src = src + self.drop_path(self.dropout2(src2))                                                             # Dropout
        return src


class TransformerClassifier(Module):  # Multi Layer Perceptron
    def __init__(self,
                 seq_pool=True,                                                     # True per CVT e CCT
                 embedding_dim=256,                                                 # dimensione data in ingresso
                 num_layers=12,                                                     # layers
                 num_heads=12,                                                      # head
                 mlp_ratio=4.0,                                                     # niente di nuovo
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None):
        super().__init__()
        # controlla il positional
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        
        dim_feedforward = int(embedding_dim * mlp_ratio)                        # int 128 * 1 o 2 = 128 o 256
        self.embedding_dim = embedding_dim                                      # dim 128
        self.sequence_length = sequence_length                                  # 3136
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:            # controlla il sequence pool, per ViT non serve
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)  # 1 1 128
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)     # questo per ViT

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':                 # per ViT e CVT
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim), requires_grad=True) # zeri 1 3 128
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim), requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)                   # init dropout
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)                                                             # Layer Normalization

        #self.linear = Linear(embedding_dim, num_classes)        # passaggio finale, Linear, da fare con tutti e 3 ViT, CVT e CCT, non dovrebbe servirci
        self.apply(self.init_weight)

    def forward(self, x):     # x è quello che esce dal tokenizer
        # input x = 32 16384 128

        if self.positional_emb is None and x.size(1) < self.sequence_length:                    # entra subito dopo l'init di self.positional
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        # CLASS TOKEN for ViT
        if not self.seq_pool: 
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            # x = 32 16385 128  #aggiunge un token

        # POSITIONAL EMBEDDING
        if self.positional_emb is not None:         # entra se non è 0, all'init sono tutti 0   # PROBLEMA QUI
            # shape of self.positional_emb = 1 16385 128
            x += self.positional_emb # 

        x = self.dropout(x)       # 32 16385 128                                                      # Dropout


        ### START Transformer Encoder
        for blk in self.blocks:   # per ogni blocco (layer) esegue il TranformerEncoder
            x = blk(x)
        x = self.norm(x)    # Layer Normalization

        ### END Transformer Encoder

        if self.seq_pool:   # Sequence pooling, da fare solo con CVT e CCT
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)                         # softmax
        else:
            x = x[:, 0]         # slice the array, taking all rows (;) but keeping the first column (1), è il flatten??
        # [batch_size, features_dim]
        #print(x.shape)
        #x = self.linear(x)      # Linear -> embedding, num_classes = fc_output_dim  [feature_dim, fc_output_dim] tolgo il linear, sta nell'aggregation
        
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

