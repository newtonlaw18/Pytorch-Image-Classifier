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

def process_image(image):
    processed_img = Image.open(image)  
    preprocess_img = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                          std=(0.229, 0.224, 0.225))
                                    ])
    
    processed_img = preprocess_img(processed_img)    
    return processed_img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()
 
    prediction_image = process_image(image_path).unsqueeze(0)     
    with torch.no_grad():
        output = model.forward(prediction_image)
        probability, labels = torch.topk(output, topk)  
        probability = probability.exp()
        
        class_idx = {model.class_to_idx[k]: k for k in model.class_to_idx}
        probable_classes = []

        for lbl in labels.numpy()[0]:
            probable_classes.append(class_idx[lbl])
    return probability.numpy()[0], probable_classes