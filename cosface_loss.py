
# Based on https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py

import torch
import torch.nn as nn
from torch.nn import Parameter

# SphereFace: devo moltiplicare m all'angolo theta
# ArcFace: devo sommare m all'angolo theta
# CosFace: devo sommare m al coseno

# dim (int) = dimensione where cosine similarity is computed. default = 1
# eps (float) = small value to avoid division by zero. default = 1e-8

def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    ip = torch.mm(x1, x2.t()) # prodotto tra x1 e x2
    w1 = torch.norm(x1, 2, dim) # x1 feature vector, L2, dim
    w2 = torch.norm(x2, 2, dim) # x2 weight vector, L2, dim
    return ip / torch.ger(w1, w2).clamp(min=eps)

# mm: performs a matrix multiplication of the matrices x1 and x2.t
# ger: outer product of w1 and w2. if w1 is a vector of size n and w2 is a vector of size m, then out must be a matrix of size (N x M)
# clamp: clamp all elements in input into a range [min, max]. In this case we have only min

# In the testing stage, the face recognition score of a testing face pair is usually calculated according to cosine similarity between the two feature vectors.
# This suggests that the norm of feature vector x is not contributing to the scoring function. Thus in training stage, we fix ||x|| = s. Consequently, the posterior
# probability merely relies on cosine of angle. 

class MarginCosineProduct(nn.Module): # CosFace
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m  # ho m una uguale per tutti e 3 i casi SphereFace, ArcFace e CosFace
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    # tensor.zeros_like() = return a tensor filled with the scalar value 0, with the same size of input
    # equivale a torch.zeros(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).
    # tensor.scatter(dim, index, src, reduce = None) = scrive tutti i valori dal tensore src dentro self agli indici specificati del tensore
    # tensor.view() = ritorna un nuovo tensor con gli stessi dati del self tensor ma con forma diversa

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight) # calcola la cosine similarity, cos(theta)
        one_hot = torch.zeros_like(cosine) # tensor di 0 con la stessa dimensione di cosine
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # dim = 1 ottiene un tensor colonna, con indici le label, valori tutti 1 che andrà a moltiplicare per m, e sottrarrà il tutto a cosine
        output = self.s * (cosine - one_hot * self.m) # perché non usa add ma si crea one_hot?
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

#################### ArcFace (ArcFace) ###############################################
# SECONDA VERSIONE GIUSTA
class ArcFace(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.50): # modificati da s = 30, m = 0.4
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    # tensor.where(condition, x, y) ritorna un tensor di element selezionati da x o y, dipendente dalla condizione
    # tensor.acos(input, *, out = None) = calcola l'inverso del coseno per ogni elemento in input
    # tensor.cos(input, *, out = None) = calcola il coseno per ogni elemento in input

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight) # calcola la cosine similarity, cos(theta)
        one_hot = torch.zeros_like(cosine) # tensor di 0 con la stessa dimensione di cosine
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        one_hot.mul_(self.m)
        cosine.acos_() # ora ho un tensor arccos di cosine
        cosine += one_hot
        cosine.cos_() # ritrasformo in cosine
        output = self.s * cosine
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


# PRIMA VERSIONE SBAGLIATA
# class ArcFace(nn.Module):
#     """Implement of large margin cosine distance:
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#     """
#     def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.40): # modificati da s = 30, m = 0.4
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
    
#     # tensor.where(condition, x, y) ritorna un tensor di element selezionati da x o y, dipendente dalla condizione
#     # tensor.acos(input, *, out = None) = calcola l'inverso del coseno per ogni elemento in input
#     # tensor.cos(input, *, out = None) = calcola il coseno per ogni elemento in input

#     def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
#         cosine = cosine_sim(inputs, self.weight) # calcola la cosine similarity, cos(theta)
#         # invece di sottrarre m al cosine, devo aggiungerlo all'angolo del cosine
#         # one_hot = torch.zeros_like(cosine)
#         # one_hot.scatter_(1, label.view(-1, 1), self.m) # 1 al posto self.m
#         cosine.acos_() # ora ho un tensor arccos di cosine
#         # se io sommo due tensori, ottengo un matrice, io non voglio questo
#         # cosine += one_hot # * self.m # sommo m
#         cosine.add(self.m) # aggiungo m all'angolo
#         cosine.cos() # ritrasformo in cosine
#         cosine.mul(self.s) # moltiplico per s
#         # output = self.s * cosine 
#         return cosine
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + 'in_features=' + str(self.in_features) \
#                + ', out_features=' + str(self.out_features) \
#                + ', s=' + str(self.s) \
#                + ', m=' + str(self.m) + ')'

#################### SphereFace (A-softmax) ###############################################

class SphereFace(nn.Module):
    """Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 1.5): # modificati da s = 30, m = 0.4
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m # per sphereface, m>=1, bisogna capire questo valore di m, vedere 3.4 del paper
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    # tensor.where(condition, x, y) ritorna un tensor di element selezionati da x o y, dipendente dalla condizione
    # tensor.acos(input, *, out = None) = calcola l'inverso del coseno per ogni elemento in input
    # tensor.cos(input, *, out = None) = calcola il coseno per ogni elemento in input

    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight) # calcola la cosine similarity, cos(theta)
        one_hot = torch.zeros_like(cosine) # tensor di 0 con la stessa dimensione di cosine
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        one_hot.mul_(self.m)
        cosine.acos_() # ora ho un tensor arccos di cosine
        cosine.mul_(one_hot) # da vedere se funziona
        cosine.cos_()# ritrasformo in cosine
        output = self.s * cosine
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


#################### ArcFace (ArcFace) ###############################################
# Ho trovato queste implementazioni di cosFace e ArcFace 
# Based on https://github.com/deepinsight/insightface/blob/master/recognition/vpl/losses.py
 
# def get_loss(name):
#     if name == "cosface":
#         return CosFace()
#     elif name == "arcface":
#         return ArcFace()
#     else:
#         raise ValueError()

# class CosFace(nn.Module):
#     def __init__(self, s=64.0, m=0.40):
#         super(CosFace, self).__init__()
#         self.s = s
#         self.m = m

#     def forward(self, cosine, label):
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#         m_hot.scatter_(1, label[index, None], self.m)
#         cosine[index] -= m_hot
#         ret = cosine * self.s
#         return ret


# class ArcFace(nn.Module):
#     def __init__(self, s=64.0, m=0.5):
#         super(ArcFace, self).__init__()
#         self.s = s
#         self.m = m

#     def forward(self, cosine: torch.Tensor, label):
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#         m_hot.scatter_(1, label[index, None], self.m)
#         cosine.acos_()
#         cosine[index] += m_hot
#         cosine.cos_().mul_(self.s)
#         return cosine
 


#################### SphereFace (A-softmax) ###############################################
# Ho trovato queste implementazioni di SphereFace
# Based on https://github.com/clcarwin/sphereface_pytorch/blob/master/net_sphere.py
 
# class AngleLoss(nn.Module):
#     def __init__(self, gamma=0):
#         super(AngleLoss, self).__init__()
#         self.gamma   = gamma
#         self.it = 0
#         self.LambdaMin = 5.0
#         self.LambdaMax = 1500.0
#         self.lamb = 1500.0
 
#     def forward(self, input, target):
#         self.it += 1
#         cos_theta,phi_theta = input
#         target = target.view(-1,1) #size=(B,1)
 
#         index = cos_theta.data * 0.0 #size=(B,Classnum)
#         index.scatter_(1,target.data.view(-1,1),1)
#         index = index.byte()
#         index = Variable(index)
 
#         self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
#         output = cos_theta * 1.0 #size=(B,Classnum)
#         output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
#         output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)
 
#         logpt = F.log_softmax(output)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
 
#         loss = -1 * (1-pt)**self.gamma * logpt
#         loss = loss.mean()
 
#         return loss


# class AngleLinear(nn.Module):
#     def __init__(self, in_features, out_features, m = 4, phiflag=True):
#         super(AngleLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         self.phiflag = phiflag
#         self.m = m
#         self.mlambda = [
#             lambda x: x**0,
#             lambda x: x**1,
#             lambda x: 2*x**2-1,
#             lambda x: 4*x**3-3*x,
#             lambda x: 8*x**4-8*x**2+1,
#             lambda x: 16*x**5-20*x**3+5*x
#         ]

#     def forward(self, input):
#         x = input   # size=(B,F)    F is feature len
#         w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

#         ww = w.renorm(2,1,1e-5).mul(1e5)
#         xlen = x.pow(2).sum(1).pow(0.5) # size=B
#         wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

#         cos_theta = x.mm(ww) # size=(B,Classnum)
#         cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
#         cos_theta = cos_theta.clamp(-1,1)

#         if self.phiflag:
#             cos_m_theta = self.mlambda[self.m](cos_theta)
#             theta = Variable(cos_theta.data.acos())
#             k = (self.m*theta/3.14159265).floor()
#             n_one = k*0.0 - 1
#             phi_theta = (n_one**k) * cos_m_theta - 2*k
#         else:
#             theta = cos_theta.acos()
#             phi_theta = myphi(theta,self.m)
#             phi_theta = phi_theta.clamp(-1*self.m,1)

#         cos_theta = cos_theta * xlen.view(-1,1)
#         phi_theta = phi_theta * xlen.view(-1,1)
#         output = (cos_theta,phi_theta)
#         return output # size=(B,Classnum,2)


# class SphereFace(nn.Module):
#     """Implement of large margin cosine distance:
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#     """
#     def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 1.5): # modificati da s = 30, m = 0.4
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m # per sphereface, m>=1, bisogna capire questo valore di m, vedere 3.4 del paper
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
    
#     # tensor.where(condition, x, y) ritorna un tensor di element selezionati da x o y, dipendente dalla condizione
#     # tensor.acos(input, *, out = None) = calcola l'inverso del coseno per ogni elemento in input
#     # tensor.cos(input, *, out = None) = calcola il coseno per ogni elemento in input

#     def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
#         cosine = cosine_sim(inputs, self.weight) # calcola la cosine similarity, cos(theta)
#         # invece di sottrarre m al cosine, devo aggiungerlo all'angolo del cosine
#         # one_hot = torch.zeros_like(cosine) #
#         # one_hot.scatter_(1, label.view(-1, 1), 1.0)
#         cosine.acos_() # ora ho un tensor arccos di cosine
#         cosine.mul(self.m) # dovrebbe essere giusto, non ho ancora ben capito al 100% la questione del one_hot
#         cosine.cos_() # lo riporto normale
#         cosine.mul(self.s)
#         return cosine
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#                + 'in_features=' + str(self.in_features) \
#                + ', out_features=' + str(self.out_features) \
#                + ', s=' + str(self.s) \
#                + ', m=' + str(self.m) + ')'