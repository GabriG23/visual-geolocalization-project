
import torch
import logging
import torchvision
from torch import nn

from model.layers import Flatten, L2Norm, GeM, Autoencoder, Attention
# from layers import Flatten, L2Norm, GeM, Autoencoder, Attention

CHANNELS_NUM_IN_LAST_CONV = {          
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "vgg16": 512,
    }

CHANNELS_NUM_AFTER_LAYER_3 = {           
        "resnet18": 256,
        "resnet50": 1024
    }


class GeoLocalizationNet(nn.Module):                 
    def __init__(self, backbone, fc_output_dim, fm_reduction_dim, reduction, num_classes=5965):           
        super().__init__()
        backbone_after_3, layers_4, features_dim, features_map_dim = get_backbone(backbone)
        self.backbone_after_3 = backbone_after_3
        self.layers_4 = layers_4
        self.aggregation = nn.Sequential(                
                L2Norm(),                                 
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),     
                L2Norm()                                 
            )                                               
        self.attention = Attention(features_map_dim)                    
        self.dim_reduction = reduction
        if self.dim_reduction:
            self.autoencoder = Autoencoder(features_map_dim, fm_reduction_dim)         
        self.attn_classifier = nn.Linear(features_map_dim, num_classes)  

    def forward(self, x):
        feature_map = self.backbone_after_3(x)             
        
        x = self.layers_4(feature_map)
        global_features = self.aggregation(x)              
        
        feature_map = feature_map.detach()                   
        
        if self.dim_reduction:
            red_feature_map, rec_feature_map = self.autoencoder(feature_map) 
        else:
            rec_feature_map = feature_map    
            red_feature_map = feature_map       
        attn_prelogits, attn_scores = self.attention(feature_map, rec_feature_map)

        attn_logits = self.attn_classifier(attn_prelogits) 
        return global_features, attn_logits, feature_map, rec_feature_map, red_feature_map, attn_scores


def get_backbone(backbone_name):                         
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)    
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone_name == "resnet152":
            backbone = torchvision.models.resnet152(pretrained=True)
        
        for name, child in backbone.named_children():               
            if name == "layer3": 
                break                                           
            for params in child.parameters():                     
                params.requires_grad = False                       
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        backbone_after_3 = list(backbone.children())[:-3]      
        layers_4 = list(backbone.children())[-3]                  
                                                                    
    
    elif backbone_name == "vgg16":                                
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]            
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    backbone_after_3 = torch.nn.Sequential(*backbone_after_3)         
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]                                                                             
    features_map_dim = CHANNELS_NUM_AFTER_LAYER_3[backbone_name]
    return backbone_after_3, layers_4, features_dim, features_map_dim


