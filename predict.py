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
import argparse
import model_checkpoint
import predict_image
import print_prediction_result

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', help = 'specify image path', 
                    default='./flowers/valid/88/image_00546.jpg')
parser.add_argument('--checkpoint', help = 'choose nn model checkpoint name', default='checkpoint.pth')
parser.add_argument('--top_k', help = 'set top k value', type=int, default=5)
parser.add_argument('--gpu', help = 'set gpu to use gpu computation', default="gpu")

args = parser.parse_args()
print(args)

# params for prediction
image_path = args.image_path
nn_checkpoint = args.checkpoint
top_k = args.top_k

# load nn model checkpoint
nn_checkpoint_model = model_checkpoint.load_nn_model_checkpoint(nn_checkpoint)

# start predicting image specified
probability, probable_classes = predict_image.predict(image_path, nn_checkpoint_model, top_k)

# print prediction result
print_prediction_result.print_result(probability, probable_classes, top_k)
