import torch
import logging
import torchvision
from torch import nn

from model.layers import Flatten, L2Norm, GeM
# from backbone import vit, cvt, cct
from backbone import cvt_and_cct

CHANNELS_NUM_IN_LAST_CONV = {           # questi dipendono dall'architettura della rete
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "vgg16": 512,
        "cvt": 512,
        "cct": 512
        # ,"vit": 224,
        # "cvt": 224,
        # "cct": 224
    }

class GeoLocalizationNet(nn.Module):                        # questa è la rete principale
    def __init__(self, backbone, fc_output_dim):            # l'oggetto della classe parent è creato in funzione della backbone scelta
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone, fc_output_dim)
        self.aggregation = nn.Sequential(                   # container sequenziale di layers, che sono appunto eseguiti in sequenza come una catena
                L2Norm(),                                   # questi sono le classi definite in layers
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),     # applica la trasformazione y = x @ A.T + b dove A sono i parametri della rete in quel punto 
                L2Norm()                                    # e b è il bias aggiunto se è passato bias=True al modello. I pesi e il bias sono inizializzati
            )                                               # random dalle features in ingresso

        self.linear = nn.Linear(features_dim, fc_output_dim)
        self.l2norm = L2Norm()
        self.backbone_name = backbone

        self.aggregation2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(512, fc_output_dim),
            nn.BatchNorm1d(fc_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_output_dim, fc_output_dim),
            nn.BatchNorm1d(fc_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_output_dim, fc_output_dim),
            L2Norm()
        )
    
    def forward(self, x):
        x = self.backbone(x)        # con resnet18 esce [32, 512, 7, 7]
        x = self.aggregation2(x)     # con resnet18 esce [32, 512]
        return x


def get_backbone(backbone_name, fc_output_dim):         # backbone_name è uno degli argomenti del programma
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)     # loading del modello già allenato
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone_name == "resnet152":
            backbone = torchvision.models.resnet152(pretrained=True)
        
        for name, child in backbone.named_children():               # ritorna un iteratore che permette di iterare sui moduli nella backbone
                                                                    # restituendo una tupla con nome e modulo per ogni elemento
            if name == "layer3":  # Freeze layers before conv_3
                break                                               # fa il freeze di tutti i layers precedenti in modo da non 
            for params in child.parameters():                       # perdere informazioni durante il transfer learning
                params.requires_grad = False                        # freeza i parametri del modello in modo che questi non cambino durante l'ottimizzazione   
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]                     # rimuove gli utlimi due layers della backbone (avg pooling and FC layer) in modo
                                                                    # da poterci attaccare i successivi del nuovo modello (aggregation)
        backbone = torch.nn.Sequential(*layers)                         # crea una backbone dopo la manipolazione dei layers


    elif backbone_name == "vgg16":                                  # qui fa la stessa cosa con questa backbone
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")

        backbone = torch.nn.Sequential(*layers)                         # crea una backbone dopo la manipolazione dei layers

    elif backbone_name == "cvt": # Vision Transformer Lite            CVT 224x224
        backbone = cvt_and_cct.cvt_initialization(fc_output_dim)
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]  
        return backbone, features_dim   
    elif backbone_name == "cct": # Convolutional Vision Transformer   CCT 224x224
        backbone = cvt_and_cct.cct_initiliaziation(fc_output_dim)
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]  
        return backbone, features_dim   
   
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]         # prende la dimensione corretta dell'utlimo layer in modo da poterla
                                                                    # mettere come dimensione di input per il linear layer successivo                                     
    return backbone, features_dim





            # # [32, 3, 224, 224]
            # x = self.backbone(x)                            # con transformers ritorna feature di dim [32, num_classes] [batch_size, features_dim] che deve essere uguale all'output
            # # [32, 224]
            # x = torch.unsqueeze(x, 2)  # Add a dummy spatial dimension of size 1
            # x = torch.unsqueeze(x, 3)  # Add another dummy spatial dimension of size 1
            # # [32, 224, 1, 1]
            # x = nn.functional.adaptive_avg_pool2d(x, 1)     # global averaage pooling to reduce spatial dimensions resulting in a tensor of shape [batch_size, features_dim, 1, 1].
            # # [32, 224, 1, 1]
            # # [1, 1]
            # x = x.view(x.size(0), -1)                       # This flattens the feature map obtained from the previous step into a 1D tensor. It reshapes the tensor to have a shape of [batch_size, features_dim], where each element corresponds to a specific feature channel.
            # # [32, 224]
            # # [1, 1]
            # # prima di arrivare qui deve avere [batch_size, feature_dim]
            # x = self.aggregation_vit(x)                     # This applies additional aggregation operations to the flattened feature map. In the provided code, it consists of a sequence of operations including L2 normalization, linear transformation (using nn.Linear), and another L2 normalization





# import torch
# import logging
# import torchvision
# from torch import nn

# from model.layers import Flatten, L2Norm, GeM
# from backbone import vit, cvt, cct

# CHANNELS_NUM_IN_LAST_CONV = {           # questi dipendono dall'architettura della rete
#         "resnet18": 512,
#         "resnet50": 2048,
#         "resnet101": 2048,
#         "resnet152": 2048,
#         "vgg16": 512,
#         "vit": 224,
#         "cvt": 224,
#         "cct": 224
#     }

# class GeoLocalizationNet(nn.Module):                        # questa è la rete principale
#     def __init__(self, backbone, fc_output_dim, layers):            # l'oggetto della classe parent è creato in funzione della backbone scelta
#         super().__init__()
#         self.backbone, features_dim = get_backbone(backbone, fc_output_dim, layers)
#         self.aggregation = nn.Sequential(                   # container sequenziale di layers, che sono appunto eseguiti in sequenza come una catena
#                 L2Norm(),                                   # questi sono le classi definite in layers
#                 GeM(),
#                 Flatten(),
#                 nn.Linear(features_dim, fc_output_dim),     # applica la trasformazione y = x @ A.T + b dove A sono i parametri della rete in quel punto 
#                 L2Norm()                                    # e b è il bias aggiunto se è passato bias=True al modello. I pesi e il bias sono inizializzati
#             )                                               # random dalle features in ingresso
#         self.aggregation_transformer = nn.Sequential(
#             L2Norm(),
#             nn.Linear(features_dim, fc_output_dim),
#             L2Norm()
#         )
#         self.linear = nn.Linear(features_dim, fc_output_dim)
#         self.l2norm = L2Norm()
#         self.backbone_name = backbone
    
#     def forward(self, x):
#         if self.backbone_name in ["vit", "cvt", "cct"]:
#             # [32, 3, 224, 224]
#             x = self.backbone(x)                            # con transformers ritorna feature di dim [32, num_classes] [batch_size, features_dim] che deve essere uguale all'output
#             #x = self.aggregation_transformer(x)     # con resnet18 esce [32, 512]

#         #     [32, 224]
#             x = torch.unsqueeze(x, 2)  # Add a dummy spatial dimension of size 1
#             x = torch.unsqueeze(x, 3)  # Add another dummy spatial dimension of size 1
#         #     [32, 224, 1, 1]
#             x = nn.functional.adaptive_avg_pool2d(x, 1)     # global averaage pooling to reduce spatial dimensions resulting in a tensor of shape [batch_size, features_dim, 1, 1].
#         #     [32, 224, 1, 1]
#         #     [1, 1]
#             x = x.view(x.size(0), -1)                       # This flattens the feature map obtained from the previous step into a 1D tensor. It reshapes the tensor to have a shape of [batch_size, features_dim], where each element corresponds to a specific feature channel.
#         #     [32, 224]
#         #     [1, 1]
#         #     prima di arrivare qui deve avere [batch_size, feature_dim]
#             x = self.aggregation_transformer(x)                     # This applies additional aggregation operations to the flattened feature map. In the provided code, it consists of a sequence of operations including L2 normalization, linear transformation (using nn.Linear), and another L2 normalization
#         else:
#             x = self.backbone(x)        # con resnet18 esce [32, 512, 7, 7]
#             x = self.aggregation(x)     # con resnet18 esce [32, 512]
#         return x


# def get_backbone(backbone_name, fc_output_dim, layers):         # backbone_name è uno degli argomenti del programma
#     if backbone_name.startswith("resnet"):
#         if backbone_name == "resnet18":
#             backbone = torchvision.models.resnet18(pretrained=True)     # loading del modello già allenato
#         elif backbone_name == "resnet50":
#             backbone = torchvision.models.resnet50(pretrained=True)
#         elif backbone_name == "resnet101":
#             backbone = torchvision.models.resnet101(pretrained=True)
#         elif backbone_name == "resnet152":
#             backbone = torchvision.models.resnet152(pretrained=True)
        
#         for name, child in backbone.named_children():               # ritorna un iteratore che permette di iterare sui moduli nella backbone
#                                                                     # restituendo una tupla con nome e modulo per ogni elemento
#             if name == "layer3":  # Freeze layers before conv_3
#                 break                                               # fa il freeze di tutti i layers precedenti in modo da non 
#             for params in child.parameters():                       # perdere informazioni durante il transfer learning
#                 params.requires_grad = False                        # freeza i parametri del modello in modo che questi non cambino durante l'ottimizzazione   
#         logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
#         layers = list(backbone.children())[:-2]                     # rimuove gli utlimi due layers della backbone (avg pooling and FC layer) in modo
#                                                                     # da poterci attaccare i successivi del nuovo modello (aggregation)
#         backbone = torch.nn.Sequential(*layers)                         # crea una backbone dopo la manipolazione dei layers


#     elif backbone_name == "vgg16":                                  # qui fa la stessa cosa con questa backbone
#         backbone = torchvision.models.vgg16(pretrained=True)
#         layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
#         for layer in layers[:-5]:
#             for p in layer.parameters():
#                 p.requires_grad = False
#         logging.debug("Train last layers of the VGG-16, freeze the previous ones")

#         backbone = torch.nn.Sequential(*layers)                         # crea una backbone dopo la manipolazione dei layers

#     elif backbone_name == "vit": # Vision Transformer Lite            ViT Lite 224x224 
#         features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]  
#         return vit.vision_transformer_lite(fc_output_dim, layers), features_dim            
#     elif backbone_name == "cvt": # Convolutional Vision Transformer   CVT 224x224 
#         features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]  
#         return cvt.convolutional_vision_transformer(fc_output_dim, layers), features_dim
#     elif backbone_name == "cct": # Convolutional Compact Transformer  CCT 224x224
#         features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]  
#         return cct.convolutional_compact_transformer(fc_output_dim, layers), features_dim

   
#     features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]         # prende la dimensione corretta dell'utlimo layer in modo da poterla
#                                                                     # mettere come dimensione di input per il linear layer successivo                                     
#     return backbone, features_dim

