import torch.nn as nn
from .transformers import TransformerClassifier
from .tokenizer import Tokenizer
import logging

def convolutional_compact_transformer(fc_output_dim): # embedding_dim mettere 224?
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=fc_output_dim, img_size=224)

# Compact Convolutional Trasformers: utilizes a convolutional tokenizer, generating richer toknes and preserving local information.
# The The convolutional tokenizer is better at encoding relationships between patches compared to the original ViT
# Tokenizer: ConvLayer + Pooling + Reshape
# Trasformer Classifier: Transformer Encoder + SeqPool + Linear Layer

def _cct(num_layers, num_heads, mlp_ratio, embedding_dim, img_size, kernel_size=3, stride=None, padding=None):

    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)

    padding = padding if padding is not None else max(1, (kernel_size // 2))

    model = CCT(num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                img_size=img_size
                )

    return model

class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=224,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=5965,
                 positional_embedding='learnable',
                 ):
                 
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer( n_input_channels=n_input_channels,              # canali input
                                    n_output_channels=embedding_dim,                # canali output
                                    kernel_size=kernel_size,                        # stride block  ???
                                    stride=stride,                                  # kernel block ???
                                    padding=padding,                                # padding block ???
                                    pooling_kernel_size=pooling_kernel_size,
                                    pooling_stride=pooling_stride,
                                    pooling_padding=pooling_padding,
                                    max_pool=True,
                                    activation=nn.ReLU,
                                    n_conv_layers=n_conv_layers,
                                    conv_bias=False)

        self.classifier = TransformerClassifier(
                                    sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels, height=img_size, width=img_size),
                                    embedding_dim=embedding_dim,
                                    seq_pool=True,
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
