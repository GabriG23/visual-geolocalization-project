# Based on https://github.com/ydwen/opensphere

import torch
import torch.nn as nn
from torch.nn import Parameter

#################### ArcFace (ArcFace) ###############################################

# dim (int) = dimensione where cosine similarity is computed. default = 1
# eps (float) = small value to avoid division by zero. default = 1e-8

# la cosine similarity è un modo per calcolare cos(theta), ed è uguale a cos(theta) = A*B / (||A||*||B||) dove A e B sono due vettori di attributi
def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    # in x1 feature vector, in x2 weight vector
    ip = torch.mm(x1, x2.t())                        # prodotto matriciale tra feature e weight, in questo caso i weight vengono trasposti (perché non hanno la stessa dimensione?)
    w1 = torch.norm(x1, 2, dim)                      # normalizza il vettore feature, 2 = L2 norm. Ritorna una matrice norm o un vettore norm di un tensore
                                                     #  dim serve per specificare quale dimensione/i del tensor di input calcolare (in questo caso dim = 1)
    w2 = torch.norm(x2, 2, dim)                      # normalizza il vettore weight
    return ip / torch.ger(w1, w2).clamp(min=eps)  # nel clamp metto solo il valore minimo 1e-8

# mm: esegue un prodotto matriciale torch.mm(input, mat2,*, out=None)
# ger: outer product of w1 and w2. if w1 is a vector of size n and w2 is a vector of size m, then out must be a matrix of size (N x M)
# clamp: clamp all elements in input into a range [min, max]. In this case we have only min

class ArcFace(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30, m: float = 0.3): # m >= 0  provato con s = 64 e m = 0.5, e  s = 30 e m = 0.4
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m   # valore del margine
        self.weight = Parameter(torch.Tensor(out_features, in_features))  # Il parameter è una sotto classe del tensore, quindi qui crea due sottoclassi con le out_features e le in_feature
        nn.init.xavier_uniform_(self.weight) # riempie il tensor con valori in base al tipo di distribuzione, in questo caso Xavier_uniform
    
    # tensor.zeros_like() = return a tensor filled with the scalar value 0, with the same size of input
    # equivale a torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).
    # tensor.scatter(dim, index, src, reduce = None) = scrive tutti i valori dal tensore src dentro self agli indici specificati del tensore
    # tensor.view() = ritorna un nuovo tensor con gli stessi dati del self tensor ma con forma diversa
    
    # tensor.acos(input, *, out = None) = calcola l'inverso del coseno per ogni elemento in input
    # tensor.cos(input, *, out = None) = calcola il coseno per ogni elemento in input
    # tensor.no_grad():  disabilità il grandiente. Context-manager that disabled gradient calculation. Disabling gradient calculation is useful for inference, when you are sure that you will not call tensor.backward().
    #                   It will reduce memory consuption for computations that would otherwise have requires_grad=True

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight) # calcola la cosine similarity, cos(theta), vedi sopra

        with torch.no_grad():
            theta_m = torch.acos(cosine.clamp(-1+1e-5, 1-1e-5)) # calcolo l'arcocoseno del tensor e lo clampo tra -1 e 1
            theta_m.scatter_(1, label.view(-1, 1), self.m, reduce='add') # stessa cosa di cosplace, somma self.m
            theta_m.clamp_(1e-5, 3.14159) # clampo tra 0 e pigreco
            d_theta = torch.cos(theta_m) - cosine # riporto al coseno e sottraggo il cosine. Non capisco perché faccia così, qui sottrae e sotto risomma cosine

        logits = self.s * (cosine + d_theta) # al cosine originale somma il d_theta calcolato
        # loss = F.cross_entropy(logits, label) # non serve, noi la facciamo fuori
        # return loss
        return logits
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
