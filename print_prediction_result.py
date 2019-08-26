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

def print_result(probability, probable_classes, top_k):
    cat_to_name = category_label.load_json_data()
    flower_classes = [cat_to_name[name] for name in probable_classes]    
    print('\n---------- Top 5 Classes Predicted on the image ----------\n')
    print('Highest Probability Predicted Flower Class: {}\n'.format(flower_classes[0]))
    for i in range(top_k):
        print("{:.6f}% of {}".format(probability[i] * 100, [cat_to_name[name] for name in probable_classes][i]))
    print('\n')
      