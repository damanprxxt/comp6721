
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

# variables

epochs = 1
learning_rate = 0.001

print(f"epochs = {epochs}, learning_rate= {learning_rate}")

# reading our datasets
datastore = os.path.abspath("Data_set\DS_New")

dataset = pd.DataFrame()

dir = ["newestdataset_new/without_mask",
       "newestdataset_new/surgical_mask",
       "newestdataset_new/cloth_mask",
       "newestdataset_new/with_N95_mask",
       "newestdataset_new/with_N95_mask_valve"]
for i in range(0, len(dir)):
    for path in os.listdir(dir[i]):
        full_path = os.path.abspath(dir[i])
        dataset = dataset.append({'img': str(full_path + "\\\\" + path), 'groups': i}, ignore_index=True)

stored_object_name = 'newestdataset_new/dataset.pickle'
print(f'Saving Dataframe to: {stored_object_name}')
dataset.to_pickle(stored_object_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# image transforming and resizing
class detect_mask(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

        self.transformations = Compose([
            Resize((128, 128)),
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


# CNN model
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        #[(W−K+2P)/S]+1
        # W is the input volume
        # K is the Kernel size 
        # P is the padding 
        # S is the stride 
        #128
        #(3,128,128)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        #(32*128*128)
        self.pool = nn.MaxPool2d(2,2)
        #(32*64*64)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        #(32*64*64)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #(64*64*64)
        self.pool = nn.MaxPool2d(2,2)
        #(64*32*32)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        #(64*16*16)
        self.fc1 = nn.Linear(64*16*16, 5)

    def forward(self, input):
        output = nn.functional.relu(self.bn1(self.conv1(input)))
        output = self.pool(output)
        output = nn.functional.relu(self.bn2(self.conv2(output)))                              
        output = nn.functional.relu(self.bn3(self.conv3(output)))
        output = self.pool(output)
        output = nn.functional.relu(self.bn4(self.conv4(output)))
        output = self.pool(output)
        #flatenning the output
        output = output.view(-1,64*16*16)
        output = self.fc1(output)

        return output

# =============================================================================
# class MyCNN(nn.Module):
#     def __init__(self):
#         super(MyCNN, self).__init__()
#         # [(W−K+2P)/S]+1
#         # W is the input volume
#         # K is the Kernel size
#         # P is the padding
#         # S is the stride
#         # 32
#         # (3,32,32)
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(12)
#         # 32-5+2+1=32
#         # (20*32*32)
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(20)
#         self.pool = nn.MaxPool2d(2, 2)
#         # 16
#         # (20*16*16)
#         # 16-3+2+1=16
#         # 32*16*16
#         self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(32)
#         # 16-3+2+1=16
#         # op=32
#         # (32*16*16)
#         self.fc1 = nn.Linear(32 * 16 * 16, 5)
# 
#     def forward(self, input):
#         output = nn.functional.relu(self.bn1(self.conv1(input)))
#         output = nn.functional.relu(self.bn2(self.conv2(output)))
#         output = self.pool(output)
#         output = nn.functional.relu(self.bn3(self.conv3(output)))
#         output = output.view(-1, 32 * 16 * 16)
#         output = self.fc1(output)
# 
#         return output
# 
# =============================================================================

# plot
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


def data_prepration(path) -> None:
    mask_df = pd.read_pickle(path)
    # print the distribution
    print(mask_df['groups'].value_counts())
    train, test = train_test_split(mask_df, test_size=0.2, random_state=0,
                                       stratify=mask_df['groups'])
    return [
        detect_mask(train),
        detect_mask(test),
        nn.CrossEntropyLoss()
    ]


def train_dataloader(train_df) -> DataLoader:
    return DataLoader(train_df, batch_size=32, shuffle=True, num_workers=0)


def val_dataloader(validate_df) -> DataLoader:
    return DataLoader(validate_df, batch_size=32, num_workers=0)


train_df, validate_df, cross_entropy_loss = data_prepration("newestdataset_new/dataset.pickle")


def model_train():
    optimizer = Adam(CNN_model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        epoch_loss = 0.0
        train_loader = train_dataloader(train_df)
        for i, data in enumerate(train_loader, 0):
            images, target = data['img'], data['groups']
            images, target = images.to(device), target.to(device)
            labels = target.flatten()
            outputs = CNN_model(images)
            batch_loss = cross_entropy_loss(outputs, labels)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        print(f' Training loss at {epoch}:', epoch_loss)


def evaluation():
    predictions, actuals = torch.tensor([]), torch.tensor([])
    train_loader = val_dataloader(validate_df)
    for i, data in enumerate(train_loader):
        images, targets = data['img'], data['groups']
        if device.type == 'cuda':
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


CNN_model = MyCNN().to(device)

# printing the summary
print(summary(CNN_model, input_size=(3, 128, 128)))

#model_train()
#print('model has been trained')
#torch.save(CNN_model.state_dict(), "train_model.pth")
#evaluation()

# =============================================================================
#
# models_path="D:\comp6721\groupProject\mainproject\model1.pth"
# torch.save(CNN_model.state_dict(), models_path)
#
# # =============================================================================
# model = Model()
# # =============================================================================
# model.load_state_dict(torch.load(models_path))
# # =============================================================================
# # model = torch.load(models_path)
# # =============================================================================
# print("model_loaded")
#
# import torchvision.transforms as transforms
# image_transformations = transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                mean=[0.5, 0.5, 0.5],
#                std=[0.5, 0.5, 0.5]
#             )
#         ])
# classes = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
# def classify (model, image_transforms, image_path, classes):
#
#     image = Image.open(image_path)
#     image= image_transforms(image).float()
#     image= image.unsqueeze(0)
#     output = model(image)
#     _, predicted = torch.max (output.data, 1)
#     print (classes[predicted.item()])
#
# classify(model,image_transformations,"New_Datasets\with_N95_mask\\N95_mask (1).png",classes)
#
# =============================================================================
# =============================================================================
# rand_sampler
# def predict(model):
#     rand_sampler = torch.utils.data.RandomSampler(validate_df, num_samples=32, replacement=True)
#     print(rand_sampler)
#     data = iter(DataLoader(validate_df, batch_size=32, num_workers=0, sampler=rand_sampler)).next()
#     inputs,targets = data['image'], data['mask']
#     inputs = inputs.cuda()
#     targets = targets.cuda()
#     output = model(inputs)
#     output = torch.argmax(output,axis=1)
#
# print(rand_sampler)
#
#
#
#     print(data['path'][rand_ind])
#     img = Image.open(data['path'][rand_ind])
#     plt.imshow(np.asarray(img))
#     print("Actual:", class_mapping[targets[rand_ind].tolist()[0]])
#     print("Predicted:",class_mapping[output[rand_ind].tolist()])
#
# =============================================================================

from sklearn.model_selection import KFold

from pathlib import Path

from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose(
    [transforms.Resize((128, 128)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datasetPath = Path('newestdataset_new')
dataset_name = datasetPath
dataset = datasets.ImageFolder(('./%s/') % dataset_name,
                               transform=transform)


def evaluate_phase(true_class, predicted_class, labels):
    print(classification_report(true_class, predicted_class, digits=2, target_names=labels))
    confusion_mat = confusion_matrix(true_class, predicted_class)
    groups = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
    conf_mat_plot(confusion_mat, groups)
    #confusionmatrix_dataframe = pd.DataFrame(confusion_matcted_crix(true_class, predilass), index=labels,
    #                                         columns=labels)
    return true_class, predicted_class

k_folds = 1
num_epochs = 1
loss_fx = nn.CrossEntropyLoss()
results = {}
torch.manual_seed(42)
kfold = KFold(n_splits=k_folds, shuffle=True)
print('--------------------------------')

#trueclasseslist=[]
#predictedclasseslist=[]
#traininglisseslist=[]

train, test = train_test_split(dataset, test_size=0.2, random_state=0)
for fold, (train_ids, test_ids) in enumerate(kfold.split(train)):
    print(f'FOLD= {fold}')
    print('--------------------------------')
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    trainloader = torch.utils.data.DataLoader(dataset,
        batch_size=32, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(dataset,
        batch_size=32, sampler=test_subsampler)

    minibatchsize=len(trainloader)
    Mymodel = MyCNN()
    optimizer = torch.optim.Adam(Mymodel.parameters(), lr=1e-4)
    training_losses=[] #to store the loss in training data
    for epoch in range(0, num_epochs):
        print(f'Starting epoch {epoch + 1}')
        current_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, targets = data
            optimizer.zero_grad()
            outputs = Mymodel(images)
            loss = loss_fx(outputs, targets)
            #             training_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i % minibatchsize == (minibatchsize-1):
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / minibatchsize))
                current_loss = 0.0
        print('Training process has finished. Saving trained model.')
        print('Starting testing')
        training_losses.append(loss)
    #     save_path = f'./model-fold-{fold}.pth'
    #     torch.save(Mymodel.state_dict(), save_path)
    correct, total = 0, 0
    predicted_class = []
    real_class = []
    classesList = ['Without_Mask', 'Surgical_Mask', 'Cloth_Mask', 'N95_Mask', 'Valve_Mask']
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, targets = data
            outputs = Mymodel(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class += list(predicted.numpy())
            real_class += list(targets.numpy())
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print("Conf. Matrix of fold number ", fold, ":\n")
    evaluate_phase(real_class, predicted_class, classesList)
    #     trueclasseslist.append(real_class)
    #predictedclasseslist.append(predicted_class)
    print('Accuracy of fold number %d: %d %%' % (fold, 100.0 * correct / total))
    print('--------------------------------')
    results[fold] = 100.0 * (correct / total)
    torch.save(Mymodel.state_dict(), f"newestdataset_new/{fold}model.pth")
    torch.utils.data.DataLoader(test,
            batch_size=32)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, targets = data
            outputs = Mymodel(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class += list(predicted.numpy())
            real_class += list(targets.numpy())
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print("Conf. Matrix of fold number ", fold, ":\n")
    evaluate_phase(real_class, predicted_class, classesList)
#     trueclasseslist.append(real_class)
#predictedclasseslist.append(predicted_class)
print('Accuracy of whole Model: %d %%' % ( 100.0 * correct / total))
print(f'K_FOLD CROSS_VALIDATION Outcome of {k_folds} FOLDS')
print('-------------------------------\n')
Total = 0.0
for i, val in results.items():
    print(f'Fold {i}: {val} %')
    Total += val
print(f'Avg. value: {Total / len(results.items())} %')

