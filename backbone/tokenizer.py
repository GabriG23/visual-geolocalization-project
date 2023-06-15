import torch
import torch.nn as nn
import torch.nn.functional as F

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,                                          # dimensione del blocco, kernel stride padding
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,            # dimensione del pooling layer
                 n_conv_layers=1,                                                       # full convolutional layer
                 n_input_channels=3,                                                    # canali ingresso
                 n_output_channels=64,                                                  # canali output
                 in_planes=64,                                                          # piani di ingresso
                 activation=None,                                                       # True solo per CCT, 
                 max_pool=True,                                                         # True solo per CCT, Pooling solo per cct
                 conv_bias=False):                                                      # True per vit e CVT
        super(Tokenizer, self).__init__()

        # numero di filtri
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]
        #convolutional layer
        self.conv_layers = nn.Sequential(
                *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)  # unisce terza e quarta dimensione 
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):               # dimensioni immagini B C H W
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1] # ritorna il forward di questo tensore

    def forward(self, x):
        # forward, esegue i conv_layers ed esegue un reshape
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)  # fa flatten unisce terza e quarta dimensione, traspose cambia la seconda dimensione con la terza

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
#                          come sono le nostre                      come le vorrebbe lui
#                        input             output                   input         output
#     3 = input channel
# self.conv_layers(x) -> 32 3 224 224   -> 32 3 56 56             32 3 224 224 -> 32 3 56 56
# self.flattener(x)   -> 32 56 56 56 -> 32 128 12544               32 3 56 56   -> 32 56 3136
# x.transpose         -> 32 128 12544   -> 32 16384 128           32 56 3136   -> 32 3136 56