
import torch
import logging
import torchvision
from torch import nn
import torch.nn.functional as F
from model.layers import Flatten, L2Norm, GeM, feature_L2_norm

CHANNELS_NUM_IN_LAST_CONV = {           # questi dipendono dall'architettura della rete
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "vgg16": 512,
    }

##### COSPLACE
class GeoLocalizationNet(nn.Module):                        # questa è la rete principale
    def __init__(self, backbone, fc_output_dim):            # l'oggetto della classe parent è creato in funzione della backbone scelta
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = nn.Sequential(                   # container sequenziale di layers, che sono appunto eseguiti in sequenza come una catena
                L2Norm(),                                   # questi sono le classi definite in layers
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),     # applica la trasformazione y = x @ A.T + b dove A sono i parametri della rete in quel punto 
                L2Norm()                                    # e b è il bias aggiunto se è passato bias=True al modello. I pesi e il bias sono inizializzati
            )                                               # random dalle features in ingresso
    
    
    def forward(self, x):
        x = self.backbone(x)                                # prima entra nella backbone
        x = self.aggregation(x)                             # e dopo entra nel container sequenziale
        return x

def get_backbone(backbone_name):                            # backbone_name è uno degli argomenti del programma
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
    
    elif backbone_name == "vgg16":                                  # qui fa la stessa cosa con questa backbone
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone = torch.nn.Sequential(*layers)                         # crea una backbone dopo la manipolazione dei layers
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]         # prende la dimensione corretta dell'utlimo layer in modo da poterla
                                                                    # mettere come dimensione di input per il linear layer successivo
    return backbone, features_dim


##### GEOWARP
class FeatureExtractor(nn.Module):                        # questa è la rete principale
    def __init__(self, backbone, fc_output_dim):            # l'oggetto della classe parent è creato in funzione della backbone scelta
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone)
        self.aggregation = nn.Sequential(                   # container sequenziale di layers, che sono appunto eseguiti in sequenza come una catena
                L2Norm(),                                   # questi sono le classi definite in layers
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),     # da capire se servono 
                L2Norm()                                    
            )                                               # random dalle features in ingresso
        # in più CosPlace ha Flatten, nn.linear e un'altra L2Norm
        self.avgpool = nn.AdaptiveAvgPool2d((15, 15))           # avgpool per i descrittori locali
        self.l2norm = L2Norm()                                  # norma per i descrittori locali

    def forward(self, x, f_type = "global"):
        x = self.backbone(x)                # backbone per entrambi i descrittori
        if f_type == "local":
            x = self.avgpool(x)             # per descrittori locali
            x = self.l2norm(x)
            return x
        elif f_type == "global":
            x = self.aggregation(x)         # per descrittori global # e dopo entra nel container sequenziale
            return x                 
        else:
            raise ValueError(f"Invalid features type: {f_type}")

##### MODULE GEOWARP
class HomographyRegression(nn.Module):
    def __init__(self, output_dim=16, kernel_sizes=[7, 5], channels=[225, 128, 64], padding=0):
        super().__init__()
        assert len(kernel_sizes) == len(channels) - 1, \
            f"In HomographyRegression the number of kernel_sizes must be less than channels, but you said {kernel_sizes} and {channels}"
        nn_modules = []
        for in_channels, out_channels, kernel_size in zip(channels[:-1], channels[1:], kernel_sizes):
            nn_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            nn_modules.append(nn.BatchNorm2d(out_channels))
            nn_modules.append(nn.ReLU())
        self.conv = nn.Sequential(*nn_modules)  # in ordine Conv2d, BatchNorm e ReLU
        # Find out output size of last conv, aka the input of the fully connected
        shape = self.conv(torch.ones([2, 225, 15, 15])).shape           
        output_dim_last_conv = shape[1] * shape[2] * shape[3]
        self.linear = nn.Linear(output_dim_last_conv, output_dim)
        # Initialize the weights/bias with identity transformation
        init_points = torch.tensor([-1, -1, 1, -1, 1, 1, -1, 1]).type(torch.float)
        init_points = torch.cat((init_points, init_points))
        self.linear.bias.data = init_points
        self.linear.weight.data = torch.zeros_like((self.linear.weight.data))

    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(B, 8, 2)
        return x.reshape(B, 8, 2)

##### MODULE GEOWARP
class GeoWarp(nn.Module):
    """
    Overview of the network:
    name                 input                                       output
    FeaturesExtractor:   (2B x 3 x H x W)                            (2B x 256 x 15 x 15)
    compute_similarity:  (B x 256 x 15 x 15), (B x 256 x 15 x 15)    (B x 225 x 15 x 15)
    HomographyRegression:(B x 225 x 15 x 15)                         (B x 16)
    """
    
    def __init__(self, features_extractor, homography_regression):
        super().__init__()
        self.features_extractor = features_extractor                    # rete uguale a Cosplace: backbone + aggregation pooling
        self.homography_regression = homography_regression              # rete per il warping

    def forward(self, operation, args):
        """Compute a forward pass, which can be of different types.
        This "ugly" step of passing the operation as a string has been adapted
        to allow calling different methods through the Network.forward().
        This is because only Network.forward() works on multiple GPUs when using torch.nn.DataParallel().
        
        Parameters
        ----------
        operation : str, defines the type of forward pass.
        args : contains the tensor(s) on which to apply the operation.
        
        """
        assert operation in ["features_extractor", "similarity", "regression", "similarity_and_regression"]
        if operation == "features_extractor":                                   # Encoder
            if len(args) == 2:
                tensor_images, features_type = args                             # prende il tensore e le feature dagli args
                return self.features_extractor(tensor_images, features_type)    # ritorna l'output dei descrittori globali
            else:
                tensor_images = args
                return self.features_extractor(tensor_images, "local")          # altrimenti i descrittori sono locali, quindi ritorna quello dei locali
        
        elif operation == "similarity":                                         # controlla se devo calcolare la similarity
            tensor_img_1, tensor_img_2 = args                                   # prendo i due tensori
            return self.similarity(tensor_img_1, tensor_img_2)                  # ritorno la matrice similarity
        
        elif operation == "regression":                                         # controlla se devo effettuare la regression
            similarity_matrix = args                                            # prende la matrice prodotta dalla similarity
            return self.regression(similarity_matrix)                           # ritorna la regressione
        
        elif operation == "similarity_and_regression":                          # se devo effettuare entrambe
            tensor_img_1, tensor_img_2 = args
            similarity_matrix_1to2, similarity_matrix_2to1 = self.similarity(tensor_img_1, tensor_img_2)
            return self.regression(similarity_matrix_1to2), self.regression(similarity_matrix_2to1)
    
    def similarity(self, tensor_img_1, tensor_img_2):
        features_1 = self.features_extractor(tensor_img_1.cuda(), "local")               # prende ler feature del primo tensore
        features_2 = self.features_extractor(tensor_img_2.cuda(), "local")               # prende le feature del secondo tensore
        similarity_matrix_1to2 = compute_similarity(features_1, features_2)     # la similarity ritorna due matrici, sarà poi compito del regression module W effettuare la misura di similarity
        similarity_matrix_2to1 = compute_similarity(features_2, features_1)
        return similarity_matrix_1to2, similarity_matrix_2to1
    
    def regression(self, similarity_matrix):
        return self.homography_regression(similarity_matrix)

def compute_similarity(features_a, features_b):                                  # calcola la similarity dalle immagini warpaed
    b, c, h, w = features_a.shape                                                # prende le shape Batch, Channel, Height, Weight
    features_a = features_a.transpose(2, 3).contiguous().view(b, c, h*w)         # fa una traspose, le s alva in memoria in modo contiguo e fa view
    features_b = features_b.view(b, c, h*w).transpose(1, 2)
    features_mul = torch.bmm(features_b, features_a)                             # moltiplica le due feature
    correlation_tensor = features_mul.view(b, h, w, h*w).transpose(2, 3).transpose(1, 2)
    correlation_tensor = feature_L2_norm(F.relu(correlation_tensor))
    return correlation_tensor