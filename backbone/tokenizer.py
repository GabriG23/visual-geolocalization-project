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
                 activation=None,                                                       # True solo per CCT
                 max_pool=True,                                                         # True solo per CCT
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
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1] # ritorna i canali, 3

    def forward(self, x):
        print(x.shape)
        x = self.conv_layers(x)
        print(x.shape)
        x = self.flattener(x)
        print(x.shape)
        x = x.transpose(-2, -1)
        print(x.shape)
        return x
        # return self.flattener(self.conv_layers(x)).transpose(-2, -1)  # fa flatten unisce terza e quarta dimensione, traspose cambia la seconda dimensione con la terza

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

# passi:
# x = x = torch.zeros(32, 3, 512, 512)    # B, C, H, W
#          input        output 
# conv2d   B Ci H W -> B Co Ho Wo
# identity B C H W  -> B C  H  W
# max pool B C H W  -> B C  Ho Wo
#
#
# flattener = nn.Flatten(2, 3)  # unisce terza e quarta dimensione 
# x = flattener(x)  -> shape 32, 3, 262144
# x = traspose(-2, -1) -> shape 32, 262144, 3
#
# output B HxW C da 4 a 3 dimensioni
#
#
#
#
#
#
#
#
