import torch
import torch.nn as nn

from models import resnet_scene as resnet
from transformers import DPTModel, DPTConfig, DPTImageProcessor, DPTForSemanticSegmentation, DPTForDepthEstimation
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union


class ResNet(nn.Module):
    def __init__(self, orig_resnet):
        super(ResNet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1

        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2

        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x1 = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5 
    



class Baseline(nn.Module):
    def __init__(self, cfg):
        super(Baseline, self).__init__()
        self.arch = cfg.arch
        self.semantic = cfg.semantic
        self.extra_semantic_layers = cfg.extra_semantic_layers
        if cfg.arch == 'dpt':
            self.arch = 'dpt'
            self.dpt_config = DPTConfig(image_size=256)
            self.dpt =  DPTForSemanticSegmentation(config = self.dpt_config).from_pretrained("Intel/dpt-large-ade")
            self.dpt.config.image_size = 256
            self.dpt_config = self.dpt.config
            self.dpt_head = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                # nn.Dropout(self.dpt.config.semantic_classifier_dropout),
                nn.Conv2d(256, 128, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(kernel_size=(65,1), stride = 1),
                nn.ReLU()
            )
            
        else:
            orig_resnet = resnet.__dict__[cfg.arch](pretrained=cfg.pretrained)
            self.backbone = ResNet(orig_resnet)
        self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relu = nn.ReLU(inplace=True)

        channel = 64
        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_conv5 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv4 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv3 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv2 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv1 = nn.Conv2d(channel, channel, (1, 1))
        self.up_conv0 = nn.Conv2d(channel, channel, (1, 1))

        # lateral
        self.c5_conv = nn.Conv2d(2048, channel, (1, 1))
        self.c4_conv = nn.Conv2d(1024, channel, (1, 1))
        self.c3_conv = nn.Conv2d(512, channel, (1, 1))
        self.c2_conv = nn.Conv2d(256, channel, (1, 1))
        self.c1_conv = nn.Conv2d(128, channel, (1, 1))

        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)


        

        # plane or non-plane classifier
        self.pred_prob = nn.Conv2d(channel, 1, (1, 1), padding=0)
        # embedding
        self.embedding_conv = nn.Conv2d(channel, 2, (1, 1), padding=0)
        # depth prediction
        self.pred_depth = nn.Conv2d(channel, 1, (1, 1), padding=0)
        # surface normal prediction
        self.pred_surface_normal = nn.Conv2d(channel, 3, (1, 1), padding=0)
        # surface plane parameters
        self.pred_param = nn.Conv2d(channel, 3, (1, 1), padding=0)
        
        if cfg.semantic:
            # semantic segmentation
            self.pred_semantic = nn.Conv2d(channel, 41, (1, 1), padding=0)
            if cfg.extra_semantic_layers:
                self.pred_semantic2 = nn.Conv2d(41, 2, (1, 1), padding=0)
                self.combination = nn.Conv2d(4, 2, (1, 1), padding=0)

            else: 
                # combination for semantic pool
                self.combination = nn.Conv2d(43, 2, (1, 1), padding=0)


    def top_down(self, x):
        c1, c2, c3, c4, c5 = x

        p5 = self.relu(self.c5_conv(c5))
        p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(c4))
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(c3))
        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(c2))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(c1))

        p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0, p1, p2, p3, p4, p5
    
    def dpt_backbone(self, x):
        # Get Hidden States from dpt encoder
        l1 = self.dpt.dpt(x, output_hidden_states=True, output_attentions=True, return_dict=True)
        hidden_states = l1.hidden_states
        hidden_states = [ feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.dpt.config.backbone_out_indices]
        # Pass hidden states to dpt decoder
        l2 = self.dpt.neck(hidden_states=hidden_states)
        l3 = l2[-1]
        # Pass to head
        l4 = self.dpt_head(l3)
        return l1,l2,l3,l4

    def forward(self, x):
        if self.arch == 'dpt':
            p3,p2,p1,p0 = self.dpt_backbone(x)
        else:
            # bottom up
            c1, c2, c3, c4, c5 = self.backbone(x)
            # top down
            p0, p1, p2, p3, p4, p5 = self.top_down((c1, c2, c3, c4, c5))
            

        prob = self.pred_prob(p0)
        embedding = self.embedding_conv(p0)
        depth = self.pred_depth(p0)
        surface_normal = self.pred_surface_normal(p0)
        param = self.pred_param(p0)
        
        if self.semantic:
            semantic = self.pred_semantic(p0)
            if self.extra_semantic_layers:
                semantic2 = self.pred_semantic2(semantic)
                combination = self.combination(torch.cat((embedding, semantic2), dim=1))   
            else:
                combination = self.combination(torch.cat((embedding, semantic), dim=1))
            return prob, embedding, depth, surface_normal, param, semantic, combination
            
        return prob, embedding, depth, surface_normal, param

    
