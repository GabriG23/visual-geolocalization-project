
# Based on https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py

import torch
import torch.nn as nn
from torch.nn import Parameter


def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t())           # mopltiplica i descrittori con i pesi del layer. .t() restituisce il vettore così com'è se mono-dimensionale
    w1 = torch.norm(x1, 2, dim)         # ritorna la norma euclidea del tensore calcolata sulla dimensione 1 (il descrittore penso sia un vettore riga)
    w2 = torch.norm(x2, 2, dim)         # stesso discorso di su ma con i pesi
    return ip / torch.ger(w1, w2).clamp(min=eps)        

    # A.T@b = ||A||*||b||*cos(theta)
    # cos(theta) = A.T@b / (||A||*||b||)

    # x1 = (32, 518)                descrittore
    # x2 = (5965, 518)              pesi del layer
    # ip = (32, 5965)               prodotto matriciale
    # w1 = (32)
    # w2 = (5965)
    # return (32, 5965)

    # "we reformulate the softmax loss as a cosine loss by L2 normalizing both features 
    # and weight vectors to remove radial variations, based on which a cosine margin  
    # term m is introduced to further maximize the decision margin in the angular space"

    # QUINDI il margine serve per ulteriormente massimizzare il decision margin all'interno dello spazio angolare in cui avviene la proiezione
    # un margine angolare è preferito ad uno Euclideo

class MarginCosineProduct(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.40):
        super().__init__()
        self.in_features = in_features                  # dimensione dei descrittori
        self.out_features = out_features                # numero di classi del gruppo
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))    # i parametri sono costituiti da un tesnore di questa forma
        nn.init.xavier_uniform_(self.weight)                    # inizilizza i parametri del modello con una distribuzione uniforme
    
    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:       # prende i descrittori delle immagini e le corrispettive labels
        cosine = cosine_sim(inputs, self.weight)                # calcola la cosine similarity a partire dai descrittori e dai pesi
        one_hot = torch.zeros_like(cosine)                      # crea un tensore composto da zeri con le stesse proprietà di cosine
        one_hot.scatter_(1, label.view(-1, 1), 1.0)             # label diventa un tensore colonna (con -1 prende la seconda dimensione)
                                                                # mette l'uno nella colonna specificata dalla riga di label.view(-1, 1)
        output = self.s * (cosine - one_hot * self.m)           # riduce solo dei punti specifici della matrice cosine del valore del margine
        return output       # toglire il margine m porta ad allontanare la classe corretta dalla altre tramite il one_hot
                            # cos(theta1) - m > cos(theta2) rende il vincolo più stringente per la classificazione


    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

