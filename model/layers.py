
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)      
        self.eps = eps                      
    
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps) 
    
    def __repr__(self):         
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"    

class Flatten(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"  
        return x[:, :, 0, 0]

class L2Norm(nn.Module):           
    def __init__(self, dim=1):      
        super().__init__()
        self.dim = dim              
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)     
                                                       

class Attention(nn.Module):
    def __init__(self, input_features):                   
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, 512, 1, 1, padding='same')                                                                            
        self.bn = nn.BatchNorm2d(512)                                                                                                                            
        self.relu = nn.ReLU()           
        self.conv2 = nn.Conv2d(512, 1, 1, 1, padding='same')           
        self.softplus = nn.Softplus()   

    
    def forward(self, x, rec_feature_map=None):   
        input = x                                         
        x = self.conv1(x)                                
        x = self.bn(x)                                         
        x = self.relu(x)                                   

        score = self.conv2(x)                               
        prob = self.softplus(score)                        

        if rec_feature_map is None:                        
            rec_feature_map = input   
        
        rec_feature_map_norm = F.normalize(rec_feature_map, p=2, dim=1) 
        att = torch.mul(rec_feature_map_norm, prob)        

        feat = torch.mean(att, [2, 3])                         
        return feat, prob        
    
class Autoencoder(nn.Module):
    def __init__(self, input_features, output_features):   
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, 1, 1, padding='same')    
        self.conv2 = nn.Conv2d(output_features, input_features, 1, 1, padding='same')    
        self.relu = nn.ReLU()                   

    def forward(self, x):                                  
        reduced_dim = self.conv1(x)                   
        x = self.conv2(reduced_dim)                      
        expanded_dim = self.relu(x)           
        return reduced_dim, expanded_dim


