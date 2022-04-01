from cgi import test
from imghdr import tests
from sklearn.utils import shuffle
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader
from TNT import imshow

BATCH_SIZE = 4
EPOCH = 20
#On Mac
#TESTDIR = "../Dataset/data/SNP_small/test"
#On Win
TESTDIR = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test"


data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
testset = datasets.ImageFolder(TESTDIR,transform=data_transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
classes = testset.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataiter = iter(testloader)
images, labels = dataiter.next()




model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512,5)

#PATH = "./model_ft_SNP_small.pth"
#model2 = models.resnet18()
#for param in model2.parameters():
#    param.requires_grad = False
#model2.fc = nn.Linear(512,5)
#model2.load_state_dict(torch.load(PATH))
