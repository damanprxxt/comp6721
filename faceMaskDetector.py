
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
from biasCalculator import gender_bias,gender_bias_trained_model,age_bias,age_bias_trained_model



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


    
#CNN model
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
    


#plot
def conf_mat_plot(cmatrix, groups, normalize=False, heading='CONFUSION MATRIX', type_map=matplt.cm.viridis):
    matplt.clf()
    matplt.imshow(cmatrix, cmap=type_map)
    lbl = np.arange(len(groups))
    matplt.xticks(lbl, groups, rotation=60)
    matplt.title(heading)
    matplt.colorbar()
    matplt.yticks(lbl, groups)

    print(cmatrix)

    thold = cmatrix.max() / 4.
    for a, b in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        matplt.text(b, a, cmatrix[a, b],
                 horizontalalignment="center",
                 color="blue" if cmatrix[a, b] > thold else "yellow")

    matplt.tight_layout()
    matplt.ylabel('Actual label')
    matplt.xlabel('Predicted label')

def data_prepration_bias(path1,path2) -> None:
    mask_df_male = pd.read_pickle(path1)
    mask_df_female = pd.read_pickle(path2)
    # print the distribution
    # print(mask_df_['groups'].value_counts())
    train_male, test_male = train_test_split(mask_df_male, test_size=0.2, random_state=42,
                                       stratify=mask_df_male['groups'])
    train_female, test_female = train_test_split(mask_df_female, test_size=0.2, random_state=42,
                                       stratify=mask_df_female['groups'])
    
    test_male_name = f'gender_pickle_files/test_male.pickle'
    test_female_name = f'gender_pickle_files/test_female.pickle'
    train_combined_name = f'gender_pickle_files/train_combined.pickle'
    
    print('saving training and testing datasets for bias testing')
    
    frames=[train_male,train_female]
    train=pd.concat(frames)
    train=train.sample(frac=1)
    # print(train)
    test_male.to_pickle(test_male_name)
    test_female.to_pickle(test_female_name)
    train.to_pickle(train_combined_name)
    return [
        detect_mask(train),
        detect_mask(test_male),
        detect_mask(test_female),
        nn.CrossEntropyLoss()
    ]

def data_prepration_bias_age(path1,path2) -> None:
    mask_df_male = pd.read_pickle(path1)
    mask_df_female = pd.read_pickle(path2)
    # print the distribution
    # print(mask_df_['groups'].value_counts())
    train_male, test_male = train_test_split(mask_df_male, test_size=0.2, random_state=42,
                                       stratify=mask_df_male['groups'])
    train_female, test_female = train_test_split(mask_df_female, test_size=0.2, random_state=42,
                                       stratify=mask_df_female['groups'])
    
    test_male_name = f'age_pickle_files/test_young.pickle'
    test_female_name = f'age_pickle_files/test_adult.pickle'
    train_combined_name = f'age_pickle_files/train_combined.pickle'
    
    print('saving training and testing datasets for bias testing')
    
    frames=[train_male,train_female]
    train=pd.concat(frames)
    train=train.sample(frac=1)
    # print(train)
    test_male.to_pickle(test_male_name)
    test_female.to_pickle(test_female_name)
    train.to_pickle(train_combined_name)
    return [
        detect_mask(train),
        detect_mask(test_male),
        detect_mask(test_female),
        nn.CrossEntropyLoss()
    ]

def data_prepration(path1,path2) -> None:
    mask_df_male = pd.read_pickle(path1)
    mask_df_female = pd.read_pickle(path2)
    # print the distribution
    # print(mask_df_['groups'].value_counts())
    train_male, test_male = train_test_split(mask_df_male, test_size=0.2, random_state=42,
                                       stratify=mask_df_male['groups'])
    train_female, test_female = train_test_split(mask_df_female, test_size=0.2, random_state=42,
                                       stratify=mask_df_female['groups'])
    
    frames=[train_male,train_female]
    train=pd.concat(frames)
    train=train.sample(frac=1)
    
    # print(train)
    frames=[]
    frames=[test_male,test_female]
    test=pd.concat(frames)
    test=test.sample(frac=1)

    return [
        detect_mask(train),
        detect_mask(test),
        nn.CrossEntropyLoss()
    ]


def train_dataloader(train_df) -> DataLoader:
    return DataLoader(train_df, batch_size=64, shuffle=True, num_workers=0)


def val_dataloader(validate_df) -> DataLoader:
    return DataLoader(validate_df, batch_size=64, num_workers=0)


def model_train(epochs,CNN_model,train_df,cross_entropy_loss,models_path):
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer =Adam(CNN_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        epoch_loss = 0.0
        train_loader=train_dataloader(train_df)
        for i, data in enumerate(train_loader, 0):
            images, target = data['img'], data['groups']
            images, target = images.to(device),target.to(device)
            labels = target.flatten()
            outputs = CNN_model(images)
            batch_loss = cross_entropy_loss(outputs, labels)
            optimizer.zero_grad() 
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        print(f' Training loss at {epoch}:',epoch_loss)
    save_model(CNN_model,models_path)
    return cross_entropy_loss,CNN_model

def model_train_old(epochs,CNN_model,train_df,cross_entropy_loss,models_path):
    learning_rate = 0.001
    losses=[]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer =Adam(CNN_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        epoch_loss = 0.0
        train_loader=train_dataloader(train_df)
        for i, data in enumerate(train_loader, 0):
            images, target = data['img'], data['groups']
            images, target = images.to(device),target.to(device)
            labels = target.flatten()
            outputs = CNN_model(images)
            batch_loss = cross_entropy_loss(outputs, labels)
            optimizer.zero_grad() 
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        print(f' Training loss at {epoch}:',epoch_loss)
        losses.append(epoch_loss)
    save_model(CNN_model,models_path)
    return CNN_model,cross_entropy_loss,losses

def model_train_temp(epochs,CNN_model,train_df,cross_entropy_loss,models_path):
    learning_rate = 0.001
    losses=[]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer =Adam(CNN_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        epoch_loss = 0.0
        train_loader=train_dataloader(train_df)
        for i, data in enumerate(train_loader, 0):
            images, target = data['img'], data['groups']
            images, target = images.to(device),target.to(device)
            labels = target.flatten()
            outputs = CNN_model(images)
            batch_loss = cross_entropy_loss(outputs, labels)
            optimizer.zero_grad() 
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        print(f' Training loss at {epoch}:',epoch_loss)
        losses.append(epoch_loss)
    save_model(CNN_model,models_path)
    return CNN_model,cross_entropy_loss,losses


def evaluation(CNN_model,validate_df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions, actuals = torch.tensor([]), torch.tensor([])
    train_loader=val_dataloader(validate_df)
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
        predictions = torch.cat((predictions, output.flatten()), dim=0)
        actuals = torch.cat((actuals, targets), dim=0)
        # print metrics
    groups = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
    print(classification_report(actuals.cpu(), predictions.cpu(), digits=2, target_names=groups))
    confusion_mat = confusion_matrix(actuals.cpu().numpy(), predictions.cpu().numpy())
    conf_mat_plot(confusion_mat, groups)


def save_model(CNN_model,models_path):
    torch.save(CNN_model.state_dict(), models_path)
    print("model_Saved")

#save_model(models_path)  

def load_model(models_path):
    model=MyCNN().to(device)
    model.load_state_dict(torch.load(models_path)).to(device)
    model.eval()
    print("model_loaded")
    return model


def evaluate_phase(true_class, predicted_class, labels):
    print(classification_report(true_class, predicted_class, digits=2, target_names=labels))
    confusion_mat = confusion_matrix(true_class, predicted_class)
    groups = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
    conf_mat_plot(confusion_mat, groups)
    return true_class, predicted_class

  


def gender_bias_calculate(models_path):
    
    epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # train_male_df, test_male_df, cross_entropy_loss_both = data_prepration("gender_pickle_files/male.pickle")
    # train_female_df,test_female_df,cross_entropy_loss_both=data_prepration("gender_pickle_files/female.pickle")
    train, test_male,test_female,cross_entropy_loss_both = data_prepration_bias("gender_pickle_files/male.pickle","gender_pickle_files/female.pickle")
    CNN_model = MyCNN().to(device)
    # loss,CNN_model=model_train(epochs,CNN_model,train_male_df,cross_entropy_loss_both,models_path)
    mpath="C:\\Users\daman\Documents\CONCORDIA\SEMESTER 2\submission2\gender_pickle_files\model1.pth"
    loss,CNN_model=model_train(epochs,CNN_model,train,cross_entropy_loss_both,models_path)
    torch.save(CNN_model.state_dict(),mpath)
    gender_bias(mpath,test_male,test_female)
    # printing the summary
    print(summary(CNN_model, input_size=(3, 32, 32)))
    
    
def age_bias_calculate(models_path):
    
    epochs = 1
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # train_male_df, test_male_df, cross_entropy_loss_both = data_prepration("gender_pickle_files/male.pickle")
    # train_female_df,test_female_df,cross_entropy_loss_both=data_prepration("gender_pickle_files/female.pickle")
    mpath="C:\\Users\daman\Documents\CONCORDIA\SEMESTER 2\submission2\\age_pickle_files\model1.pth"
    train, test_male,test_female,cross_entropy_loss_both = data_prepration_bias_age("age_pickle_files/young.pickle","age_pickle_files/adult.pickle")
    CNN_model = MyCNN().to(device)
    # loss,CNN_model=model_train(epochs,CNN_model,train_male_df,cross_entropy_loss_both,models_path)
    loss,CNN_model=model_train(epochs,CNN_model,train,cross_entropy_loss_both,mpath)
   
    age_bias(mpath,test_male,test_female)
    # printing the summary
    torch.save(CNN_model.state_dict(),mpath)
    print(summary(CNN_model, input_size=(3, 32, 32)))
    
    
if __name__ == "__main__":
    # model_train()
    # print('model has been trained')

    # evaluation()
    # train, test_male,test_female,cross_entropy_loss = data_prepration_bias("gender_pickle_files/male.pickle","gender_pickle_files/female.pickle")
    # train_df=detect_mask(pd.read_pickle("C:/Users/daman/Documents/CONCORDIA/SEMESTER 2/submission2/gender_pickle_files/train_combined.pickle"))
    # CNN_model = MyCNN()
    # save_model(CNN_model,"C:\\Users\daman\Documents\CONCORDIA\SEMESTER 2\submission2\gender_pickle_files\model1.pth")
    # model_train_temp(20,CNN_model,train_df,cross_entropy_loss,"C:/Users/daman/Documents/CONCORDIA/SEMESTER 2/submission2/gender_pickle_files")
    
    "to generate new pickle files from scratch"
    # from pickleFileGenerator import generate_new_files
    # generate_new_files()
    
    
    # models_path="C:\\Users\daman\Documents\CONCORDIA\SEMESTER 2\submission2\model1.pth"
    # train, test_male,test_female,cross_entropy_loss = data_prepration_bias("gender_pickle_files/male.pickle","gender_pickle_files/female.pickle")
   
    
   
    "to test and train the old model and check the accuracy"
    # train,test,cross_entropy_loss=data_prepration("gender_pickle_files/male.pickle","gender_pickle_files/female.pickle")
    # CNN_model = MyCNN()
    # CNN_model,cross_entropy_loss,losses=model_train_old(20,CNN_model,train,cross_entropy_loss,models_path)
    # evaluation(CNN_model,test)
    
    
    "train and get GENDER_bias"
    # gender_bias_calculate(models_path)
    
    
    "train and get AGE_bias"
    # age_bias_calculate(models_path)
    
    
    "to check the bias on the already trained old model from a saved test and train set"
    # gender_bias_trained_model(models_path,'gender_pickle_files/test_male.pickle','gender_pickle_files/test_female.pickle')
    
    "to check the bias on the already trained old model from a saved test and train set"
    # age_bias_trained_model(models_path,'age_pickle_files/test_young.pickle','age_pickle_files/test_adult.pickle')
    
    "to plot the training losses after training the model"
    # matplt.plot(losses,label='Training loss')
    # matplt.legend()
    # matplt.show()

    # from kfold_training import k_fold_model
    # k_fold_model(4,MyCNN())








