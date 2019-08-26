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

# create nn model function
def create_nn_model(arch = 'vgg16', hidden_unit = 600, 
                 learning_rate = 0.001, training_mode = 'gpu'):
    
    print(arch)
    cat_to_name = category_label.load_json_data()    
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_size = model.classifier[0].in_features 
    elif arch == 'alexnet':
#         print('alexnet = true')        
        model = models.alexnet(pretrained = True)
        input_size = 9216
    else:
        print("{} is not a correct architecture. Please choose either vgg16 or alexnet.".format(arch))

    for param in model.parameters():
        param.requires_grad = False

#     print(input_size)
    output_size = len(cat_to_name)
    hidden_layer1 = 1600
    hidden_layer2 = hidden_unit
    epoch_total = 5
    drop = 0.2
    
    classifier = nn.Sequential(nn.Linear(input_size, hidden_layer1),
                      nn.ReLU(),
                      nn.Dropout(p = 0.15),
                      nn.Linear(hidden_layer1, hidden_layer2),
                      nn.ReLU(),
                      nn.Dropout(p = 0.15),
                      nn.Linear(hidden_layer2, output_size),
                      nn.LogSoftmax(dim = 1))

    model.classifier = classifier   
    print_every = 50

    if torch.cuda.is_available():
        if training_mode == 'gpu': 
            model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)   
    return model, optimizer, criterion

# train nn model function
def train_nn_model(model, optimizer, criterion, train_data_loader,
                valid_data_loader, epochs=5, training_mode='gpu'):    
    print("Initiate Model Training... \n")   
    running_loss = 0
    epoch_count = 0
    steps = 0
    print_every = 50
    valid_size = len(valid_data_loader)

    for epoch_count in range(epochs):
        for images, labels in iter(train_data_loader):
            steps = steps + 1         
            if torch.cuda.is_available():
                if training_mode == 'gpu':
                    images = images.to('cuda')
                    labels = labels.to('cuda')
            
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss = running_loss + loss.item()
            
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss, accuracy = validation_check(model, criterion, training_mode, valid_data_loader)
                print("Epoch= {} of {},".format(epoch_count + 1, epochs),
                      "Loss= {:.6f},".format(running_loss / print_every),
                      "Validation Set Loss= {:.6f},".format(validation_loss / valid_size),
                      "Validation Set Accuracy= {:.6f}%".format(accuracy / valid_size)
                     )
                running_loss = 0
                model.train()
    print("Model Training Completed Successfully.")
   
# check and validate the nn model
def validation_check(model, criterion, training_mode, valid_loader):
    test_loss, test_accuracy = 0, 0
    for data in iter(valid_loader):      
        if torch.cuda.is_available() and training_mode == 'gpu':
            images = data[0].to('cuda')
            labels = data[1].to('cuda')
        
        output = model.forward(images)
        test_loss = test_loss + criterion(output, labels).item()
        probability = torch.exp(output) 
        prediction = (labels.data == probability.max(dim = 1)[1]) 
        test_accuracy = test_accuracy + prediction.type(torch.FloatTensor).mean()
    return test_loss, test_accuracy*100
  
