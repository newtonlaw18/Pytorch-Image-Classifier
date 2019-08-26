# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from PIL import Image
import json
import category_label

# function to save nn model checkpoint
def save_nn_model_checkpoint(model, args, optimizer, train_datasets):
    checkpoint = {'epochs': args.epochs,
              'arch': args.arch,
              'learning_rate': args.learning_rate,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'optimizer_dict': optimizer.state_dict(),
              'class_to_idx': train_datasets.class_to_idx,
             }
    torch.save(checkpoint, args.save_dir)  
    print('\n--------- NN Model ----------- \n')
    print(model)
    print("\nNeural Network Model checkpoint successfully saved. \n")
    
# load nn model checkpoint   
def load_nn_model_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("{} is not a correct architecture. Please choose either vgg16 or alexnet.".format(arch))
        
    model.classifier = checkpoint['classifier']
    model.learning_rate = checkpoint['learning_rate']
    model.state_dict = checkpoint['state_dict']
    model.optimizer_dict = checkpoint['optimizer_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model