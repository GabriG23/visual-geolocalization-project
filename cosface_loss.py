
# Based on https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py

import torch
import torch.nn as nn
from torch.nn import Parameter

# value of m
# m1 (Sf), m2 (Af), m3 (Cf)
#   x    |    0    |    0
#   1    |    x    |    0
#   1    |    0    |    x

# SphereFace: devo moltiplicare m1 all'angolo theta
# ArcFace: devo sommare m2 all'angolo theta
# CosFace: devo sommare m3 al coseno. Già fatto, non serve implementazione

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
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = cosine_sim(inputs, self.weight) # calcola la cosine similarity
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = self.s * (cosine - one_hot * self.m) # cosface, 0, 0
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


#################### ArcFace (ArcFace) ###############################################

# Based on https://github.com/deepinsight/insightface/blob/master/recognition/vpl/losses.py
# DA SISTEMARE
class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine

#################### SphereFace (A-softmax) ###############################################
# la versione originale è in tensorFlow, non si capisce niente dal codice
# Based on https://github.com/clcarwin/sphereface_pytorch/blob/master/net_sphere.py

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


# QUARTA SOLUZIONE, FARE UN'UNICA FUNZIONE 


