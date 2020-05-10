import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from resources.plotcm import plot_confusion_matrix
from itertools import product
torch.set_grad_enabled(True)

transform = transforms.ToTensor()

test_set = datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=False
    ,download=True
    ,transform = transform)
train_sampler = SubsetRandomSampler(list(range(48000)))
valid_sampler = SubsetRandomSampler(list(range(12000)))

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.reshape(-1, 12*4*4)))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

#Load model
PATH = '/Users/makhanov/Yandex.Disk.localized/github_repo/CNN_pytorch/model1.pth'
model = torch.load(PATH)
model.eval()

all_preds = []
targets = []
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
for batch in test_loader:
    images, labels = batch

    preds = model(images)
    loss = F.cross_entropy(preds, labels)

    all_preds.append(torch.max(preds, dim=1).indices)
    targets.append(labels.data)
all_preds = torch.cat(all_preds)
targets = torch.cat(targets)
cm = confusion_matrix(targets, all_preds)
accuracy = accuracy_score(targets, all_preds)

print("Confusion_matrix: \n", cm)
print("Overall accuracy on test set: ", accuracy)
