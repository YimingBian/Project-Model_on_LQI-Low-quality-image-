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
from TNT import test_model

BATCH_SIZE = 4

TESTDIR = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test_ori"

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(512,5)

test_model(model,TESTDIR,BATCH_SIZE)