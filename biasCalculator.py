# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:39:24 2022

@author: daman
"""

import os
import pandas as pd
from PIL import Image
import numpy as np
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
import torch.nn as nn
from torchsummary import summary
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
import itertools
import matplotlib.pyplot as matplt
import warnings
from sklearn.metrics import classification_report, confusion_matrix ,ConfusionMatrixDisplay
# from faceMaskDetector import detect_mask



dataset = pd.DataFrame()

#image transforming and resizing
class detect_mask(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

        self.transformations = Compose([
            Resize((32, 32)),
            ToTensor(),
            Normalize(
               mean=[0.5, 0.5, 0.5],
               std=[0.5, 0.5, 0.5]
            )
        ])
    def __len__(self):
        return len(self.dataFrame.index)

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('error: slicing not possible')
        val1 = self.dataFrame.iloc[key]
        val2 = Image.open(val1['img']).convert('RGB')
        return {
            'img': self.transformations(val2),
            'groups': tensor([val1['groups']], dtype=long),
            'path': val1['img']
        }


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        #[(W−K+2P)/S]+1
        # W is the input volume
        # K is the Kernel size 
        # P is the padding 
        # S is the stride 
        #64
        #64-3+2+1= 64
        #(3,32,32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        #32-5+2+1=32
        #(20*32*32)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2,2)
        #16
        #(20*16*16)
        #16-3+2+1=16
        #32*16*16
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        #16-3+2+1=16
        #op=32
        #(32*16*16)
        self.fc1 = nn.Linear(32*16*16, 5)

    def forward(self, input):
        output = nn.functional.relu(self.bn1(self.conv1(input)))      
        output = nn.functional.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                            
        output = nn.functional.relu(self.bn3(self.conv3(output)))     
        output = output.view(-1, 32*16*16)
        output = self.fc1(output)

        return output

def data_prepration(path) -> None:
    mask_df = pd.read_pickle(path)
    # print the distribution
    print(mask_df['groups'].value_counts())
    train, validate = train_test_split(mask_df, test_size=0.25, random_state=0,
                                       stratify=mask_df['groups'])
    return [
        detect_mask(train),
        detect_mask(validate),
        nn.CrossEntropyLoss()
    ]
def conf_mat_plot(cmatrix, groups, head,normalize=False, heading="", type_map=matplt.cm.viridis):
    # matplt.imshow(cmatrix, cmap=type_map)
    # lbl = np.arange(len(groups))
    # matplt.xticks(lbl, groups, rotation=60)
    # matplt.title(heading)
    # matplt.colorbar()
    # matplt.yticks(lbl, groups)
    
    # print(cmatrix)

    # thold = cmatrix.max() / 4.
    # for a, b in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
    #     matplt.text(b, a, cmatrix[a, b],
    #              horizontalalignment="center",
    #              color="blue" if cmatrix[a, b] > thold else "yellow")
    disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix,display_labels=np.arange(len(groups)))
    disp.plot()
    matplt.title(head)
    matplt.tight_layout()
    matplt.xticks(np.arange(len(groups)), groups, rotation=90)
    matplt.yticks(np.arange(len(groups)), groups, rotation=0)
    # matplt.xticks(lbl, groups, rotation=60)
    matplt.ylabel('Actual label')
    matplt.xlabel('Predicted label')
    matplt.show()
    print(cmatrix)
    
def val_dataloader(validate_df) -> DataLoader:
    return DataLoader(validate_df, batch_size=64, num_workers=0)

def load_model(models_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MyCNN().to(device)
    model.load_state_dict(torch.load(models_path))
    model.eval()
    print("model_loaded")
    return model


#our freshly trained model can be used to calculate the bias as 
#if we use an old model there will be an ambiguity between the old training
# model and the testinf datasets as we dont want to mix each other 
def gender_bias(models_path,male_df,female_df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nCalculating Gender Bias")
    print(models_path)
    CNN_model = load_model(models_path)
    print("loaded")
    for j in range(0,2):
        var=""
        predictions, actuals = torch.tensor([]), torch.tensor([])
        if j==0:
            print("Male_bias")
            var="MALE CONFUSION MATRIX"
            train_loader=val_dataloader(male_df)
        else:
            print("female_bias")
            var="FEMALE CONFUSION MATRIX"
            train_loader=val_dataloader(female_df)
        len(train_loader)    
        for i, data in enumerate(train_loader):
            images, targets = data['img'], data['groups']
            if device.type=='cuda':
                images = images.cuda()
                targets = targets.cuda()
            targets = targets.flatten()
            output = CNN_model(images)
            output = torch.argmax(output, axis=1)
            if device.type == 'cuda':
                predictions = predictions.cuda()
                actuals = actuals.cuda()
            predictions = torch.cat((predictions, output.flatten()))
            actuals = torch.cat((actuals, targets))

        # print metrics
        groups = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
        print(classification_report(actuals.cpu(), predictions.cpu(), digits=2, target_names=groups))
        confusion_mat = confusion_matrix(actuals.cpu().numpy(), predictions.cpu().numpy())
        conf_mat_plot(confusion_mat, groups,var)

#every time a bias is calculated the testing and the training images are stored separately 
#to check the bias from the already trained model we have this function
def gender_bias_trained_model(models_path,male_pickle_path,female_pickle_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nCalculating Gender Bias from saved model")
    CNN_model = load_model("C:\\Users\daman\Documents\CONCORDIA\SEMESTER 2\submission2\gender_pickle_files\model1.pth").to(device)
    male_df = pd.read_pickle(male_pickle_path)
    female_df = pd.read_pickle(female_pickle_path)
    male_df=detect_mask(male_df)
    female_df=detect_mask(female_df)
    for j in range(0,2):
        var=""
        predictions, actuals=[],[]
        predictions, actuals = torch.tensor([]), torch.tensor([])
        if j==0:
            print("Male_bias")
            var="MALE CONFUSION MATRIX"
            train_loader=val_dataloader(male_df)
        else:
            print("female_bias")
            var="FEMALE CONFUSION MATRIX"
            train_loader=val_dataloader(female_df)
        len(train_loader)    
        for i, data in enumerate(train_loader):
            images, targets = data['img'], data['groups']
            if device.type=='cuda':
                images = images.cuda()
                targets = targets.cuda()
            targets = targets.flatten()
            output = CNN_model(images)
            output = torch.argmax(output, axis=1)
            if device.type == 'cuda':
                predictions = predictions.cuda()
                actuals = actuals.cuda()
            predictions = torch.cat((predictions, output.flatten()))
            actuals = torch.cat((actuals, targets))

        # print metrics
        groups = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
        print(classification_report(actuals.cpu(), predictions.cpu(), digits=2, target_names=groups))
        confusion_mat = confusion_matrix(actuals.cpu().numpy(), predictions.cpu().numpy())
        conf_mat_plot(confusion_mat, groups,var)

def age_bias(models_path,male_df,female_df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nCalculating age Bias")
    print(models_path)
    CNN_model = load_model(models_path)
    print("loaded")
    for j in range(0,2):
        var=""
        predictions, actuals = torch.tensor([]), torch.tensor([])
        if j==0:
            print("YOUNG_bias")
            var="YOUNG CONFUSION MATRIX"
            train_loader=val_dataloader(male_df)
        else:
            print("ADULT_bias")
            var="ADULT CONFUSION MATRIX"
            train_loader=val_dataloader(female_df)
        len(train_loader)    
        for i, data in enumerate(train_loader):
            images, targets = data['img'], data['groups']
            if device.type=='cuda':
                images = images.cuda()
                targets = targets.cuda()
            targets = targets.flatten()
            output = CNN_model(images)
            output = torch.argmax(output, axis=1)
            if device.type == 'cuda':
                predictions = predictions.cuda()
                actuals = actuals.cuda()
            predictions = torch.cat((predictions, output.flatten()))
            actuals = torch.cat((actuals, targets))

        # print metrics
        groups = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
        print(classification_report(actuals.cpu(), predictions.cpu(), digits=2, target_names=groups))
        confusion_mat = confusion_matrix(actuals.cpu().numpy(), predictions.cpu().numpy())
        conf_mat_plot(confusion_mat, groups,var)

#every time a bias is calculated the testing and the training images are stored separately 
#to check the bias from the already trained model we have this function
def age_bias_trained_model(models_path,male_pickle_path,female_pickle_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nCalculating AGE Bias from saved model")
    CNN_model = load_model(models_path).to(device)
    male_df = pd.read_pickle(male_pickle_path)
    female_df = pd.read_pickle(female_pickle_path)
    male_df=detect_mask(male_df)
    female_df=detect_mask(female_df)
    for j in range(0,2):
        var=""
        predictions, actuals=[],[]
        predictions, actuals = torch.tensor([]), torch.tensor([])
        if j==0:
            print("YOUNG_bias")
            var="YOUNG CONFUSION MATRIX"
            train_loader=val_dataloader(male_df)
        else:
            print("ADULT_bias")
            var="ADULT CONFUSION MATRIX"
            train_loader=val_dataloader(female_df)
        len(train_loader)    
        for i, data in enumerate(train_loader):
            images, targets = data['img'], data['groups']
            if device.type=='cuda':
                images = images.cuda()
                targets = targets.cuda()
            targets = targets.flatten()
            output = CNN_model(images)
            output = torch.argmax(output, axis=1)
            if device.type == 'cuda':
                predictions = predictions.cuda()
                actuals = actuals.cuda()
            predictions = torch.cat((predictions, output.flatten()))
            actuals = torch.cat((actuals, targets))

        # print metrics
        groups = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
        print(classification_report(actuals.cpu(), predictions.cpu(), digits=2, target_names=groups))
        confusion_mat = confusion_matrix(actuals.cpu().numpy(), predictions.cpu().numpy())
        conf_mat_plot(confusion_mat, groups,var)
