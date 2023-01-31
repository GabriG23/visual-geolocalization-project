
import torch
import logging
import torchvision
from torch import nn

# from model.layers import Flatten, L2Norm, GeM, Autoencoder, Attention
from layers import Flatten, L2Norm, GeM, Autoencoder, Attention



CHANNELS_NUM_IN_LAST_CONV = {           # questi dipendono dall'architettura della rete
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "vgg16": 512,
    }


class GeoLocalizationNet(nn.Module):                        # questa è la rete principale
    def __init__(self, backbone, fc_output_dim, num_classes = 5965, dim_reduction=True):            # l'oggetto della classe parent è creato in funzione della backbone scelta
        super().__init__()
        backbone_until_3, layers_4, features_dim = get_backbone(backbone)
        self.backbone_until_3 = backbone_until_3
        self.layers_4 = layers_4
        self.aggregation = nn.Sequential(                   # container sequenziale di layers, che sono appunto eseguiti in sequenza come una catena
                L2Norm(),                                   # questi sono le classi definite in layers
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),     # applica la trasformazione y = x @ A.T + b dove A sono i parametri della rete in quel punto 
                L2Norm()                                    # e b è il bias aggiunto se è passato bias=True al modello. I pesi e il bias sono inizializzati
            )                                               # random dalle features in ingresso
        self.attention = Attention(256)                     # 256 sono i canali di feature map (B, 256, 32, 32)
        self.dim_reduction = dim_reduction
        if self.dim_reduction:
            self.autoencoder = Autoencoder(256, 64)         # entrano che sono 256, quella ridotta mettiamo a 32 cosi da mantenere lo stesso rapporto del paper (8)
                                                            # EVENTUALMENTE DA PROVARE SENZA AUTOENCODER QUINDI SENZA RIDURRE LE FEATURES
        self.attn_classifier = nn.Linear(256, num_classes)  # è il numero di descrottori locali

    def forward(self, x):
        feature_map = self.backbone_until_3(x)              # prima entra nella backbone
        
        x = self.layers_4(feature_map)
        
        global_features = self.aggregation(x)               # in realtà per le global features mancherebbe il cosFace
        
        feature_map = feature_map.detach()                  # separa il tensore dal computational graph ritornando un nuovo tensore che non richiede un gradiente
        # feature_map.requires_grad = False                 # è la stessa cosa? Evita al gradiente di non tornare indietro?
        
        if self.dim_reduction:
            reduced_dim, rec_feature_map = self.autoencoder(feature_map)                    # non so a cosa serva il primo
        attn_prelogits, attn_scores, att = self.attention(feature_map, rec_feature_map)
        
        attn_logits = self.attn_classifier(attn_prelogits)
        print(reduced_dim.shape)
        return global_features, attn_logits, feature_map, rec_feature_map, attn_prelogits


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
        layers_until_3 = list(backbone.children())[:-3]             # questo non è freezato, ragionare su sta cosa
        layers_4 = list(backbone.children())[-3]                    # rimuove gli utlimi due layers della backbone (avg pooling and FC layer) in modo
                                                                    # da poterci attaccare i successivi del nuovo modello (aggregation)
    
    elif backbone_name == "vgg16":                                  # qui fa la stessa cosa con questa backbone
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]            # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone_until_3 = torch.nn.Sequential(*layers_until_3)         # backbone fino al layer 3, necessaria per recuperare la feature map
    # backbone = torch.nn.Sequential(*layers)                         # crea una backbone dopo la manipolazione dei layers
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]         # prende la dimensione corretta dell'utlimo layer in modo da poterla
                                                                    # mettere come dimensione di input per il linear layer successivo
    return backbone_until_3, layers_4, features_dim


# model = GeoLocalizationNet('resnet18', 512)

# lr = 00.1
# criterion = torch.nn.CrossEntropyLoss() 
# criterion_MSE = torch.nn.MSELoss() 
# backbone_parameters = [*model.backbone_until_3.parameters(), *model.layers_4.parameters(), *model.aggregation.parameters()]      
# model_optimizer = torch.optim.Adam(backbone_parameters, lr=lr)      # utilizza l'algoritmo Adam per l'ottimizzazione


# attention_parameters = [*model.attn_classifier.parameters(), *model.attention.parameters()]
# attention_optimizer = torch.optim.Adam(attention_parameters, lr=lr)
# autoencoder_optimizer = torch.optim.Adam(model.autoencoder.parameters(), lr=lr)

# classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]         # il classifier è dato dalla loss(dimensione descrittore, numero di classi nel gruppo) 
# classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers] 

# targets = torch.rand([2, 10])

# descriptors, attn_logits, feature_map, rec_feature_map = model(images)

# global_loss = criterion(output, targets)                                           # calcola la loss (in funzione di output e target)

# attn_loss = criterion(attn_logits, targets)
# rec_loss = criterion_MSE(rec_feature_map, feature_map)
image = torch.rand([1, 3, 512, 512])
model = GeoLocalizationNet('resnet18', 512)
t_att = model(image)
# print(model)
# print(model.layers_4.parameters())
# print(model.backbone_until_3.parameters())
# print(model.aggregation.parameters())
# print(model.attention.parameters())
# print(model.autoencoder.parameters())
# parameters_to_optimize = model.layers_4.parameters()

# backbone_parameters = [*model.backbone_until_3.parameters(), *model.layers_4.parameters(), *model.aggregation.parameters()]                                             
# model_optimizer = torch.optim.Adam(backbone_parameters, lr=.001)     