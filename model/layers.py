
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        # clamp() -> taglia tutti gli elementi tra [min, max] in questo caso solo per min
        # size() -> restituisce un oggetto di classe torch.Size con le dimensioni del tensore
        # avg_pool2d() -> applica una average-pooling 2D lungo l'input tensor in base alla dimensione del kernel (singolo numero o tupla)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)     # parametri del layer composti da un tensore di dimensione 1 moltiplicato per 3     
        self.eps = eps                      
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)   # esegue il forward eseguendo la funzione gem che prende gli attributi dell'oggetto
    
    def __repr__(self):                         # metodo speciale usato per rappresentare l'oggetto di una classe come una stringa
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"    # serve a stampare l'oggetto GeM


class Flatten(nn.Module):                       # override della classe flatten
    def __init__(self):
        super().__init__()                      # restituisce un oggetto della classe parent, cioè Module
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"  # si assicura che il tensore abbia la terza e quarta dimensione uguale ad 1
        return x[:, :, 0, 0]


class L2Norm(nn.Module):                        # least square error
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim                          # dimensione a cui va ridotto
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)      # divide il vettore lungo ogni dimensione del tensore per la norma (in questo caso euclidea perché p=2)
                                                        # di base ogni vettore è elevato a p e la loro somma è elevata ad 1/p

# Questo serve a GeoWarp
def feature_L2_norm(feature):
    epsilon = 1e-6
    # torch.pow(feature, 2) : Calcola la potenza di 2 delle feature
    # torch.sum(torch.pow(feature, 2), 1): somma ogni riga nella dimensione 1
    # torch.pow(torch.sum(torch.pow(feature, 2), 1)+epsilon, 0.5): rifa la potenza a 0.5 sommando epsilon
    # unsqueeze(1): ritorna un tensore di dimensione 1 (vettore colonna)
    # expand_us(feature): lo espande alla dimensione originale di feature?
    # contiguous: return a contiguois in memory tensor contenente gli stessi dati di inizio (per velocizzare il tutto??)
    # div: divide la feature per la norma
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1)+epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature.contiguous(), norm)

