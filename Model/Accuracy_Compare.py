
import torch.nn as nn
import torch
from torchvision import models
from TNT import test_single_model, test_single_pretrained_model

TESTDIRS = ["D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test_ori",
           "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/ORI",
           "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test",
           "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/val",
           "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP/train",
           "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP/val"   
] 

PATH1 = "./model_ft_ori_small.pth"
MODEL_NAME1 = 'TL_model'

PATH2 = "./model_ft_SNP_small.pth"
MODEL_NAME2 = 'Retrained_model'

MODEL_NAME3 = 'Pre-trained_model'

model1 = models.resnet18()
for param in model1.parameters():
    param.requires_grad = False
model1.fc = nn.Linear(512,5)
model1.load_state_dict(torch.load(PATH1))

model2 = models.resnet18()
for param in model2.parameters():
    param.requires_grad = False
model2.fc = nn.Linear(512,5)
model2.load_state_dict(torch.load(PATH2))

model3 = models.resnet18(pretrained=True)

for TESTDIR in TESTDIRS:
    test_single_model(model1, TESTDIR, True, MODEL_NAME1)
    test_single_model(model2, TESTDIR, True, MODEL_NAME2)
    test_single_pretrained_model(model3, TESTDIR, True, MODEL_NAME3)
