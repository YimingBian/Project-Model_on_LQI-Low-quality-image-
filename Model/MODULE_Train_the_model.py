from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import os
from MODULE_TNT import train_model
from MODULE_MEAN_STD import Mean_and_std_of_dataset

NUM_EPOCH = 25
BATCH_SIZE = 4
STORE_PATH = "./Results/Re-trained_models"

#cudnn.benchmark = True


# MODE: FE: feature extractor / FT: fine tunining
def Train_model(DATADIR, MODEL_NAME, OUTDIM = 5, MODE = 'FE'):
    datadir = DATADIR
    train_dir = os.path.join(datadir,'/','train')
    val_dir = os.path.join(datadir,'/','val')
    train_mean, train_var = Mean_and_std_of_dataset(train_dir,BATCH_SIZE)
    val_mean, val_var = Mean_and_std_of_dataset(val_dir,BATCH_SIZE)
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_var)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(val_mean, val_var)
    ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(datadir+ '/', x), data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("GPU is not used. Exiting the training process.\n")
        exit
    #rebuild the model
    model = models.resnet18(pretrained=True)

    if MODE == 'FE': # feature extractor
        for param in model.parameters():
            param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, OUTDIM)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    SUBPATH = f'{STORE_PATH/{MODEL_NAME}}'
    retrained_model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, NUM_EPOCH, SUBPATH)

    file1 = f'{SUBPATH}/{MODEL_NAME}/.pth'
    torch.save(retrained_model.state_dict(), file1)

    return retrained_model




#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context


