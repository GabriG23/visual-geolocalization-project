
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        # clamp() -> taglia tutti gli elementi tra [min, max] in questo caso solo per min
        # size() -> restituisce un oggetto di classe torch.Size con le dimensioni del tensore
        # avg_pool2d() -> applica una average-pooling 2D lungo l'input tensor in base alla dimensione del kernel (singolo numero o tupla)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)   # parametri del layer composti da un tensore di dimensione 1 moltiplicato per 3     
        self.eps = eps                      
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)   # esegue il farward eseguendo la funzione gem che prende gli attributi dell'oggetto
    
    def __repr__(self):         # metodo speciale usato per rappresentare l'oggetto di una classe come una stringa
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"    # serve a stampare l'oggetto GeM


class Flatten(nn.Module):       # override della classe flatten
    def __init__(self):
        super().__init__()      # restituisce un oggetto della classe parent, cioè Module
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"  # si assicura che il tensore abbia la terza e quarta dimensione uguale ad 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):            # least square error
    def __init__(self, dim=1):      # sulla colonna
        super().__init__()
        self.dim = dim              # dimensione a cui va ridotto
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)      # divide ogni dimensione del tensore per la norma (in questo caso euclidea perché p=2)
                                                        # di base ogni vettore è elevato a p e la loro somma è elevata ad 1/p

class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    '''
    def __init__(self, in_c, act_fn='relu'):            # in_c?
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)              
        self.bn = nn.BatchNorm2d(512, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

        for conv in [self.conv1, self.conv2]: 
            conv.apply(net.init_weights)

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm
        return x, att_score
    
    def __repr__(self):
        return self.__class__.__name__



p = torch.ones(1)*3     # ritorna un tensore composto di uno della dimensione passata per parametro (poi *3)
                        # quindi è un tensore sulla CPU di dimensione 1 che contiene solo il valore 3 (in particolare 3., float32)

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(p.data)
print(p)

gem_obj = GeM()
print(gem_obj)

flat = Flatten()
shape = (3, 2, 1, 2)    # quando definiamo la shape di un tensore, riga e colonne delle matrici interne sono date dagli utlimi due valori
                        # della shape. Es (2, 3, 1) -> due matrici interne con tre righe ed una colonna. In più dimensioni è
                        # complicato da visualizzare.
                        # (3, 2, 3, 1) -> 3 gruppi da 2 matrici di dimensione 1 riga e 2 colonne.
                        # Possono esserci più dimensioni ma è tutta una questione di raggruppamenti
# t = torch.rand(shape)
# print(t.data)
# f = flat(t)             # flatten non necessariamente rende tutto un vettore, ma spesso trasforma in matrice, dipende dai parametri con il quale
#                         # è chiamato
# print(f)

t1 = torch.tensor(data, dtype=torch.float32)
print(type(t1.size()))
# norm_obj = L2Norm()
# tn = norm_obj(t1)
# print(tn)