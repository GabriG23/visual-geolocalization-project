
# Based on https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py

import torch
import torch.nn as nn
from torch.nn import Parameter

# dim (int) = dimensione where cosine similarity is computed. default = 1
# eps (float) = small value to avoid division by zero. default = 1e-8

# la cosine similarity è un modo per calcolare cos(theta), ed è uguale a cos(theta) = A*B / (||A||*||B||) dove A e B sono due vettori di attributi
def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    # in x1 feature vector, in x2 weight vector
    ip = torch.mm(x1, x2)   #ho tolto una transpose                     # prodotto matriciale tra feature e weight, in questo caso i weight vengono trasposti (perché non hanno la stessa dimensione?)
    w1 = torch.norm(x1, 2, dim)                      # normalizza il vettore feature, 2 = L2 norm. Ritorna una matrice norm o un vettore norm di un tensore
                                                     #  dim serve per specificare quale dimensione/i del tensor di input calcolare (in questo caso dim = 1)
    w2 = torch.norm(x2.t, 2, dim)                      # normalizza il vettore weight
    return ip / torch.ger(w1, w2).clamp(min=eps)  # nel clamp metto solo il valore minimo 1e-8

# mm: esegue un prodotto matriciale torch.mm(input, mat2,*, out=None)
# ger: outer product of w1 and w2. if w1 is a vector of size n and w2 is a vector of size m, then out must be a matrix of size (N x M)
# clamp: clamp all elements in input into a range [min, max]. In this case we have only min

class MarginCosineProduct(nn.Module): # CosFace
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.40): # m >= 0, m = 0.5, 0.4, 0.3, in_features = 224, 384, 512, out_features=lunghezza del gruppo (classi)
        super().__init__()
        self.in_features = in_features  # 512, 224 o 384
        self.out_features = out_features # numero classi
        self.s = s
        self.m = m   # valore del margine, tensore di 5965 224
        self.weight = Parameter(torch.Tensor(out_features, in_features))  # Il parameter è una sotto classe del tensore, quindi qui crea due sottoclassi con le out_features e le in_feature
        nn.init.xavier_uniform_(self.weight) # riempie il tensor con valori in base al tipo di distribuzione, in questo caso Xavier_uniform
    
    # tensor.zeros_like() = return a tensor filled with the scalar value 0, with the same size of input
    # equivale a torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).
    # tensor.scatter(dim, index, src, reduce = None) = scrive tutti i valori dal tensore src dentro self agli indici specificati del tensore
    # tensor.view() = ritorna un nuovo tensor con gli stessi dati del self tensor ma con forma diversa

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
                                                      # in input ho il feature vector
        print(inputs.shape)
        print(self.weight.shape)
        cosine = cosine_sim(inputs, self.weight)      # calcola la cosine similarity, cos(theta), vedi sopra
        one_hot = torch.zeros_like(cosine)            # crea un tensore di 0 della stessa dimensione di cosine
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # primo valore: dim = 1
        # secondovalore : indice che andrà a modificare. Qui label viene trasformata in un tensore colonna di dimensione 1 x label.size (-1, 1)
        # terzo valore: src: dati che metterà dentro one_hot, in questo caso tutti 1, volendo si può direttamente mettere self.m
        # quarto valore: reduce, qui non c'è ma volendo si può mettere il tipo di operazione, somma, sottrazione ....

        output = self.s * (cosine - one_hot * self.m) # ritorna l'output, moltiplica one_hot per il margine e lo sottrae al coseno
        # attenzione qui manca un passaggio ovvero la cross_entropy che lui fa fuori, è il "criterion" in train.py
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

#################### CosFace ##############################################################

# Based on https://github.com/ydwen/opensphere

# class CosFace(nn.Module):
#     """reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
#        reference2: <Additive Margin Softmax for Face Verification>
#     """
#     def __init__(self, feat_dim, num_class, s=64., m=0.35):
#         super(CosFace, self).__init__()
#         self.feat_dim = feat_dim
#         self.num_class = num_class
#         self.s = s
#         self.m = m
#         self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
#         nn.init.xavier_normal_(self.w)

#     def forward(self, x, y):
#         with torch.no_grad():
#             self.w.data = F.normalize(self.w.data, dim=0)

#         cos_theta = F.normalize(x, dim=1).mm(self.w)
#         with torch.no_grad():
#             d_theta = torch.zeros_like(cos_theta)
#             d_theta.scatter_(1, y.view(-1, 1), -self.m, reduce='add')

#         logits = self.s * (cos_theta + d_theta)
#         loss = F.cross_entropy(logits, y)

#         return loss