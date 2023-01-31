import torch.nn as nn
from transformers import TransformerClassifier
from tokenizer import Tokenizer

def vision_transformer_lite(type):
    if type == 2:
        return vit_2()     # num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128
    elif type == 4:
        return vit_4()     # num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128
    elif type == 6:
        return vit_6()     # num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256 
    elif type == 7:
        return vit_7()     # num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256
    elif type == 8:
        return vit_8()     # num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256


# Vision Transformers lite

class ViTLite(nn.Module):
    def __init__(self,
                 img_size=224,                               # dim immagini
                 embedding_dim=768,                          # dim feature standard,  
                 n_input_channels=3,                         # input channel
                 kernel_size=16,                             # kernel, dovrebbe essere la patch, 16
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,                              # numero di layers
                 num_heads=6,                                # numero di head
                 mlp_ratio=4.0,
                 num_classes=1000,                          
                 positional_embedding='learnable'
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
                                   activation=None,                                         # GeLU activation, qui False
                                   n_conv_layers=1,                                         # numero convolutional layer
                                   conv_bias=True)                                          # convolutional bias

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.classifier(x)
        return x


def _vit_lite(num_layers, num_heads, mlp_ratio, embedding_dim, kernel_size=4, *args, **kwargs): 
                                    
    model = ViTLite(num_layers=num_layers,                          # numero layer                   2 - 4 - 6 - 7
                    num_heads=num_heads,                            # numero head                    2 - 2 - 4 - 4
                    mlp_ratio=mlp_ratio,                            # multi layer perceptron         1 - 1 - 2 - 2
                    embedding_dim=embedding_dim,                    # dim feature                   128 - 128 - 256 - 256
                    kernel_size=kernel_size,                        # dim kernel
                    positional_embedding='learnable',               # non ho capito, ma non dovrebbe servirci       
                    *args, **kwargs)
    return model


def vit_2():
    return _vit_lite(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128)


def vit_4():
    return _vit_lite(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128)


def vit_6():
    return _vit_lite(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256)


def vit_7():
    return _vit_lite(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256)

def vit_8():
    return _vit_lite(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256)
