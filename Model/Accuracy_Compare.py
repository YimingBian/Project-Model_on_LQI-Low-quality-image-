from cgi import test
from imghdr import tests
from math import ceil
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
from TNT import test_model, compare_models

TESTDIR = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test_ori"

#load retrained model
PATH = "./model_ft_SNP_small.pth"
model1 = models.resnet18()
for param in model1.parameters():
    param.requires_grad = False
model1.fc = nn.Linear(512,5)
model1.load_state_dict(torch.load(PATH))

#load pretrained model
model2 = models.resnet18(pretrained=True)

compare_models(model1, model2, TESTDIR)
