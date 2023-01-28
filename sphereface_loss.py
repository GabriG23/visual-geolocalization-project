# Based on https://github.com/ydwen/opensphere

import torch
import torch.nn as nn
from torch.nn import Parameter
import math

#################### SphereFace (A-softmax) ###############################################

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

class SphereFace(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 3): # m >= 1, provato con s = 30 e m = 1.5 e s = 30 e m = 2
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
    
    # tensor.where(condition, x, y) ritorna un tensor di element selezionati da x o y, dipendente dalla condizione
    # tensor.acos(input, *, out = None) = calcola l'inverso del coseno per ogni elemento in input
    # tensor.cos(input, *, out = None) = calcola il coseno per ogni elemento in input

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cos_theta = cosine_sim(inputs, self.weight) # calcola la cosine similarity, cos(theta)

        with torch.no_grad():
            m_theta = torch.acos(cos_theta.clamp(-1.+1e-5, 1.-1e-5)) # calcolo l'arcocoseno del tensor e lo clampo tra -1 e 1
            m_theta.scatter_(1, label.view(-1, 1), self.m, reduce = 'multiply') # stessa cosa di cosface, qui devo moltiplicare il margine all'angolo

            # floor: return the floor of x, the largest integer less than or equal to x. if x is not a float, delegate to x.__floor__ which should return an Integral value
            # pi: contiene variabile pigreco
            # remainder: compute Computes Python’s modulus operation entrywise. The result has the same sign as the divisor other and its absolute value is less than that of other.
            # m_theta / pi, si trova l'angolo in radianti??
            k = (m_theta / math.pi).floor() # in m theta ho il mio tensore colonna con self.m moltiplicato all'angolo, cos'è k?
            sign = -2 * torch.remainder(k, 2) + 1 # (-1)**k # calcolo il segno
            phi_theta = sign * torch.cos(m_theta) - 2. * k   # moltiplico il segno per m_theta riportato normale e sottraggo -2. * k, cosa sto sottraendo?
            d_theta = phi_theta - cos_theta

        logits = self.s * (cos_theta + d_theta)
        return logits

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
