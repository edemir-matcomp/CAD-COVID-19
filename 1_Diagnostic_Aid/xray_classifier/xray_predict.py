#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms

dataset_dict = {
    'CXR14':    {'mean': 0.497, 'std': 0.087, 'num_classes': 14,
                'classes': ['atelectasis', 'cardiomegaly', 'consolidation',
                            'edema', 'effusion', 'emphysema', 'fibrosis',
                            'hernia', 'infiltration', 'mass', 'nodule',
                            'pleural_thickening', 'pneumonia','pneumothorax']},
    
    'Padchest': {'mean': 0.504, 'std': 0.076, 'num_classes': 16,
                'classes': ['atelectasis', 'cardiomegaly', 'consolidation',
                            'edema', 'effusion', 'emphysema', 'fibrosis',
                            'hernia', 'hilar enlargement', 'infiltration',
                            'mass', 'nodule', 'pleural_thickening', 'pneumonia',
                            'pneumothorax', 'support devices']},
    
    'Chexpert': {'mean': 0.497, 'std': 0.087, 'num_classes':  8,
                'classes': ['atelectasis', 'cardiomegaly', 'consolidation',
                            'edema', 'effusion', 'pneumonia', 'pneumothorax',
                            'support devices']},
    
    'VIN':      {'mean': 0.467, 'std': 0.039, 'num_classes': 13,
                'classes': [['aortic enlargement', 'atelectasis', 'calcification', 
                             'cardiomegaly', 'consolidation', 'effusion', 'ild',
                             'infiltration', 'lesion', 'lung opacity', 'nodule/mass', 
                             'pneumothorax', 'pulmonary fibrosis']]},
    
    'COVIDx':   {'mean': 1.544, 'std': 2.279, 'num_classes':  3,
                'classes': ['covid', 'normal', 'pneumonia']}
}

architectures = [
            'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 
            'resnet18', 'resnet34', 'resnet50', 'resnet152',
            'densenet121', 'densenet161', 'densenet169', 'densenet201',
            'resnext50_32x4d', 'wide_resnet50_2', 
            'resnext101_32x8d', 'wide_resnet101_2'
]

model_rootpath = 'models'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


def adapt_net(net, name, num_classes):
    
    if 'vgg' in name:
        initial_layer = net.features[0]
        net.features = nn.Sequential(nn.Conv2d(1, initial_layer.out_channels, initial_layer.kernel_size,
                                               stride=initial_layer.stride, padding=initial_layer.padding),
                                    *list(net.features.children())[1:])
        
        in_features = net.classifier[-1].in_features
        new_classifier = nn.Sequential(*list(net.classifier.children())[:-1],
                                       nn.Linear(in_features, num_classes),
                                       nn.Sigmoid())
        net.classifier = new_classifier
    
    elif 'resnet' in name or 'resnext' in name:
        initial_layer = net.conv1
        net.conv1 = nn.Conv2d(1, initial_layer.out_channels, initial_layer.kernel_size,
                              stride=initial_layer.stride, padding=initial_layer.padding)
        
        in_features = net.fc.in_features
        net.fc = nn.Sequential(nn.Linear(in_features, num_classes),
                               nn.Sigmoid())
    
    elif 'densenet' in name:
        initial_layer = net.features[0]
        net.features = nn.Sequential(nn.Conv2d(1, initial_layer.out_channels, initial_layer.kernel_size,
                                               stride=initial_layer.stride, padding=initial_layer.padding),
                                    *list(net.features.children())[1:])
        
        in_features = net.classifier.in_features
        net.classifier = nn.Sequential(nn.Linear(in_features, num_classes),
                                       nn.Sigmoid())
               
    return net


def load_sample(dataset, sample_path):
    normalize = transforms.Normalize(mean = dataset_dict[dataset]['mean'], 
                                     std  = dataset_dict[dataset]['std'])
    
    transformSequence = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        normalize
    ])
    
    img = Image.open(sample_path).convert('L')
    img = transformSequence(img)
    
    return img.unsqueeze(0)

def predict(dataset, arch, sample_path):
    
    # instancia arquitetura 
    net = getattr(models, arch)()
    net = adapt_net(net, arch, dataset_dict[dataset]['num_classes'])
    
    # carrega modelo treinado
    model_path = f'{model_rootpath}/{dataset}_{arch}.pth' 
    states = torch.load(model_path, map_location=device)
    
    net.load_state_dict(states)
    net = net.to(device)
    net.eval()
    
    # carrega imagem
    img = load_sample(dataset, sample_path)
    img = img.to(device)
    output = net(img) 
    
    return output.detach().cpu()


# In[16]:


# sample info -> img path
rootpath = '/mnt/CADCOVID/covidxxx/images_split/test/covid'
sample_path = f'{rootpath}/1-s2.0-S0929664620300449-gr2_lrg-a.png'

# model info -> architecture name and dataset used for training.
dataset = 'CXR14'
architecture = 'resnet18'
out = predict(dataset, architecture, sample_path)

for pred in range(out.shape[1]):
    print(f"{dataset_dict[dataset]['classes'][pred]}: {out[0, pred]:.4f}")

