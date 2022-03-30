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

BATCH_SIZE = 12
EPOCH = 20


model1 = models.resnet18(pretrained=True)
for param in model1.parameters():
    param.requires_grad = False
model1.fc = nn.Linear(512,5)

PATH = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/model_ft_SNP_small.pth"
model2 = models.resnet18()
for param in model2.parameters():
    param.requires_grad = False
model2.fc = nn.Linear(512,5)
model2.load_state_dict(torch.load(PATH))
#model2.eval()

# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/SNP_small'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def train_model(model, criterion, optimizer, scheduler, textdir, num_epochs=EPOCH):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    f = open(textdir, 'a')

    for epoch in range(num_epochs):
        #print(f'Epoch {epoch}/{num_epochs - 1}')
        #print('-' * 10)
        f.write(f'\nEpoch {epoch}/{num_epochs - 1}\n')
        f.write('-' * 10)
        

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            f.write(f'\n correct predicts: {running_corrects}, total sample: {dataset_sizes[phase]}')

            #print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            f.write(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    #print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    #print(f'Best val Acc: {best_acc:4f}')
    f.write(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    f.write(f'\nBest val Acc: {best_acc:4f}')
    f.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model1 = model1.to(device)
model2 = model2.to(device)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(model1.fc.parameters(), lr=0.001, momentum=0.9)
optimizer2 = optim.SGD(model2.fc.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=7, gamma=0.1)
exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=7, gamma=0.1)

output1 = f"Results/original_network_batchsize_{BATCH_SIZE}.txt"
output2 = f"Results/trained_network_batchsize_{BATCH_SIZE}.txt"


model1 = train_model(model1, criterion, optimizer1,  exp_lr_scheduler1,output1, 20)
model2 = train_model(model2, criterion, optimizer2,  exp_lr_scheduler2,output2, 20)

