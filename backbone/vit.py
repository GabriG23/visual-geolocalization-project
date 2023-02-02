import torch.nn as nn
from .transformers import TransformerClassifier
from .tokenizer import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

def vision_transformer_lite(layers = 2, img_size = 224):
    if layers == 2:
        return vit_2(layers, img_size)     # num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128
    elif layers == 4:
        return vit_4(layers, img_size)     # num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128
    elif layers == 6:
        return vit_6(layers, img_size)     # num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256 
    elif layers == 7:
        return vit_7(layers, img_size)     # num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256
    else:
        logging.info(f"ERROR number of layers. Layers cannot be equals to {layers}")


def vit_2():
    return _vit_lite(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128, img_size = 224)   # layers, attention head, Multi layer perceptron ratio, dimensione descrittori


def vit_4():
    return _vit_lite(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128, img_size = 224)


def vit_6():
    return _vit_lite(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256, img_size = 224)


def vit_7():
    return _vit_lite(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256, img_size = 224)


def _vit_lite(num_layers, num_heads, mlp_ratio, embedding_dim, img_size, kernel_size=4):      # dimensione del kernel
                                    
    model = ViTLite(num_layers=num_layers,                          # numero layer                   2 - 4 - 6 - 7 - 8
                    num_heads=num_heads,                            # numero head                    2 - 2 - 4 - 4 - 4
                    mlp_ratio=mlp_ratio,                            # multi layer perceptron         1 - 1 - 2 - 2 - 2
                    embedding_dim=embedding_dim,                    # dim feature                   128 - 128 - 256 - 256 - 256
                    kernel_size=kernel_size,                        # dim kernel
                    positional_embedding='learnable',                # dipende molto dal positional_embedding  
                    img_size=img_size  
                    )
    return model

# positional embedding: describes the location or position of an entity in a sequence so that each position is assigned a unique representation
# dropout: data that's intentionally dropped from a NN to improve processing and time to results
# Vision Transformers lite

class ViTLite(nn.Module):
    def __init__(self,
                 img_size=224,                               # dim immagini, era 224
                 embedding_dim=768,                          # dim - passata da parametro -> 128 128 256 256 256, questo è la nostra hidden size
                 n_input_channels=3,                         # input channel
                 kernel_size=16,                             # kernel, dovrebbe essere la patch, 16
                 dropout=0.,                                 # dropout del classifier
                 attention_dropout=0.1,                      # dropout dell'attention head
                 stochastic_depth=0.1,                       # profondità stocastica
                 num_layers=14,                              # numero di layers - passata da parametro  2 4 6 7 8
                 num_heads=6,                                # numero di head   - passata da parametro  2 2 4 4 4
                 mlp_ratio=4.0,                              # mlp ratio        - passata da parametro  1 1 2 2 2
                 num_classes=5965,                           # classi di cosplace
                 positional_embedding='learnable'            # learnable
                ):
        super(ViTLite, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,                       # canali input
                                   n_output_channels=embedding_dim,                         # canali output
                                   kernel_size=kernel_size,                                 # kernel block ???
                                   stride=kernel_size,                                      # stride block  ???
                                   padding=0,                                               # padding block ???
                                   max_pool=False,                                          # per cct
                                   activation=None,                                         # per cct
                                   n_conv_layers=1,                                         # numero convolutional layer
                                   conv_bias=True)                                          # convolutional bias

        self.classifier = TransformerClassifier(
            # sequence lenght canali 3, height 224, width 224: ritorna un tensore di di zero 1, 3, 224, 224
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels, height=img_size, width=img_size), # = 3
            embedding_dim=embedding_dim,                                        # dimensione come sopra
            seq_pool=False,                                                     # lo usa cvt          
            dropout=dropout,                                                    # dropout sopra
            attention_dropout=attention_dropout,                                #valori tutti sopra
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )
    
    def forward(self, x):
        # input x = 32, 3, 224, 224   B C H W
        x = self.tokenizer(x)  
        # output x = 32, 3196, 56
        x = self.classifier(x) # transformers con img 224 224 esce 32 5965  32 num classi
        return x

# classifier consists of transformer block, each including an MSHA layer e un MPL block
# applica una layer normalization, gelu activation e un dropout.
# positional embeddings can be learnable or sinusoidal.
