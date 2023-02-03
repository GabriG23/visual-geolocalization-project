
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

class Attention(nn.Module):
    def __init__(self, input_features):                     # in_c mi sembra siano 256
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, 128, 1, 1)   # attention module (in 128, così come lui riduce 1024 in 512)
                                                            # riduce solo la profondità a 512?      
                                                            # alla fine dell'attention module, devo avere solo S' (Hs, Ws), come S (32, 32)      
        self.bn = nn.BatchNorm2d(128)                       # in ingresso dovrebbe stesso discorso per 128. Per il resto è lasciato con parametri di default
                                                            # il batch ritorna la stessa shape dell'input 
                                                            # ad ogni elemento, toglie media e varianza calcolata sull'intero batch (è fatta sulla C dimension)
        self.relu = nn.ReLU()                               # nel paper è scritto "followed by ReLU"

        self.conv2 = nn.Conv2d(128, 1, 1, 1)                # filter è 1 come nel paper; 128 sono quelli che lui riceve
        self.softplus = nn.Softplus()   # use default setting.

        # for conv in [self.conv1, self.conv2]:             # per ora non esplicito nessuna inizializzazione per il conv2
        #     nn.init.xavier_uniform_(conv.weight)           

    def forward(self, x, rec_feature_map=None):   
        input = x                                           # ([32, 256, 32, 32])
        x = self.conv1(x)                                   # ([32, 128, 32, 32])
        x = self.bn(x)                                         
        x = self.relu(x)                                   

        score = self.conv2(x)                               # ([32, 1, 32, 32])                 
        prob = self.softplus(score)                        

        # Aggregate inputs if targets is None.
        if rec_feature_map is None:                         # significa che non ha usato l'autoencoder per ridurre la dimensione
            rec_feature_map = input   
        
        # L2-normalize the featuremap before pooling.
        rec_feature_map_norm = F.normalize(rec_feature_map, p=2, dim=1) 
        att = torch.mul(rec_feature_map_norm, prob)         # ([32, 256, 32, 32])  -> o comunque la dimensione di quella con più canali
        # print(f"att:{att.shape}")
        feat = torch.mean(att, [2, 3])                      # ([32, 256]) 
        # feat = tf.reduce_mean(tf.multiply(targets, prob), [1, 2], keepdims=False)         variante in tensorflow                          
        # print(f"att:{feat.shape}")
        return feat, prob, att                              # passa anche lo score ma nel model non lo usa
    
class Autoencoder(nn.Module):
    def __init__(self, input_features, output_features):    # in_c mi sembra siano 256
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, 1, 1)    # stiamo usando un autoencoder che riduce da 256 a 64 (per mantenere più o meno la proporzione del paper)
        self.conv2 = nn.Conv2d(output_features, input_features, 1, 1)    # filter è 1 come nel paper; 256 sono quelli che lui riceve
        self.relu = nn.ReLU()                               # nel paper è scritto "followed by ReLU"

    def forward(self, x):                                  
        reduced_dim = self.conv1(x)                         # ([32, 128, 32, 32])
        x = self.conv2(reduced_dim)                         # ([32, 256, 32, 32])
        expanded_dim = self.relu(x)           
        return reduced_dim, expanded_dim


# t = torch.rand([32, 256, 32, 32])
# att = Autoencoder(256, 32)
# t_att = att(t)


# p = torch.ones(1)*3     # ritorna un tensore composto di uno della dimensione passata per parametro (poi *3)
#                         # quindi è un tensore sulla CPU di dimensione 1 che contiene solo il valore 3 (in particolare 3., float32)

# data = [[1, 2],[3, 4]]
# x_data = torch.tensor(data)
# print(p.data)
# print(p)

# gem_obj = GeM()
# print(gem_obj)

# flat = Flatten()
# shape = (3, 2, 1, 2)    # quando definiamo la shape di un tensore, riga e colonne delle matrici interne sono date dagli utlimi due valori
                        # della shape. Es (2, 3, 1) -> due matrici interne con tre righe ed una colonna. In più dimensioni è
                        # complicato da visualizzare.
                        # (3, 2, 3, 1) -> 3 gruppi da 2 matrici di dimensione 1 riga e 2 colonne.
                        # Possono esserci più dimensioni ma è tutta una questione di raggruppamenti
# t = torch.rand(shape)
# print(t.data)
# f = flat(t)             # flatten non necessariamente rende tutto un vettore, ma spesso trasforma in matrice, dipende dai parametri con il quale
#                         # è chiamato
# print(f)

# t1 = torch.tensor(data, dtype=torch.float32)
# print(type(t1.size()))
# norm_obj = L2Norm()
# tn = norm_obj(t1)
# print(tn)