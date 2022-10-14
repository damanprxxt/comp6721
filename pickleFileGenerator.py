# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:59:37 2022

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
from sklearn.metrics import classification_report, confusion_matrix

def generate_new_files():
    dataset = pd.DataFrame()
    
    "TO CALCULATE THE BIAS FOR THE GENDER BASED MODEL"
    dir_male=["seperated_dataset_gender/male/without_mask",
          "seperated_dataset_gender/male/cloth_mask",
          "seperated_dataset_gender/male/surgical_mask",
          "seperated_dataset_gender/male/with_N95_mask",
          "seperated_dataset_gender/male/with_N95_valve"]
    
    dir_female=["seperated_dataset_gender/female/without_mask",
          "seperated_dataset_gender/female/cloth_mask",
          "seperated_dataset_gender/female/surgical_mask",
          "seperated_dataset_gender/female/with_N95_mask",
          "seperated_dataset_gender/female/with_N95_mask_valve"]
    
    dirs=[dir_male,dir_female]
    dirs_name=["male","female"]
    
    for j in range(0,len(dirs)):
        dataset = pd.DataFrame()
        for i in range (0,len(dirs[j])):
            for path in os.listdir(dirs[j][i]):
                full_path = os.path.abspath(dirs[j][i])
                dataset = dataset.append({'img': str(full_path + "\\\\" + path), 'groups': i}, ignore_index=True)
                # dataset=dataset.append({'img': str(full_path+"\\\\" + path),'groups': i ,'bias_group':dirs_name[j]},ignore_index=True)
            # print(dataset)
        stored_object_name = f'gender_pickle_files/{dirs_name[j]}.pickle'
        print(f'Saving Dataframe to: {stored_object_name}')
        dataset.to_pickle(stored_object_name)
    
    
    
    
    "TO CALCULATE THE BIAS FOR THE AGE BASED MODEL"
    # young
    dir_young=["seperated_dataset_age/young/without_mask",
          "seperated_dataset_age/young/cloth_mask",
          "seperated_dataset_age/young/surgical_mask",
          "seperated_dataset_age/young/with_N95_mask",
          "seperated_dataset_age/young/with_N95_mask_valve"]
    
    #adult
    dir_adult=["seperated_dataset_age/adult/without_mask",
          "seperated_dataset_age/adult/cloth_mask",
          "seperated_dataset_age/adult/surgical_mask",
          "seperated_dataset_age/adult/with_N95_mask",
          "seperated_dataset_age/adult/with_N95_mask_valve"]
    dirs=[dir_young,dir_adult]
    dirs_name=["young","adult"]
    
    for j in range(0,len(dirs)):
        dataset = pd.DataFrame()
        for i in range (0,len(dirs[j])):
            for path in os.listdir(dirs[j][i]):
                full_path = os.path.abspath(dirs[j][i])
                dataset = dataset.append({'img': str(full_path + "\\\\" + path), 'groups': i}, ignore_index=True)
                # dataset=dataset.append({'img': str(full_path+"\\\\" + path),'groups': i ,'bias_group':dirs_name[j]},ignore_index=True)
            # print(dataset)
        stored_object_name = f'age_pickle_files/{dirs_name[j]}.pickle'
        print(f'Saving Dataframe to: {stored_object_name}')
        dataset.to_pickle(stored_object_name)


generate_new_files()


