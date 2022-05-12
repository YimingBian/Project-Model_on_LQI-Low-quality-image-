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

def Train_model(DATADIR, MODEL_NAME, NETWORK_NAME, OUTDIM = 5, MODE = 'FE'):
    """
    Usage of Train_model(DATADIR, MODEL_NAME, NETWORK_NAME, OUTDIM = 5, MODE = 'FE'):
    *********************************************************************************
    1. DATADIR: the folder that contains 'train' and 'val' image folders
    2. MODEL_NAME: the name of the retrained model to be saved
    3. NETWORK_NAME: a list of pretrained networks is here "https://pytorch.org/vision/stable/models.html"
    4. OUTDIM: output dimension
    5. Two options for MODE: 'FE'-deature extractor, only train the last layer while keep the previous layers frozen. 'FT'-fine tune, train the whole network
    *********************************************************************************
    """
    datadir = DATADIR
    train_dir = f'{datadir}/train'
    val_dir = f'{datadir}/val'
    
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
    print(f'Using device: {device}')
    if device == "cpu":
        print("GPU is not used. Exiting the training process.\n")
        return None
    #rebuild the model
    #model = models.resnet18(pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0',NETWORK_NAME,pretrained = True)

    if MODE == 'FE': # feature extractor
        for param in model.parameters():
            param.requires_grad = False
    
    if NETWORK_NAME == 'resnet18':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, OUTDIM)
    elif NETWORK_NAME == 'vgg16':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, OUTDIM, bias=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    SUBPATH = f'{STORE_PATH}/{MODEL_NAME}'
    Checkpoint_model = f'{SUBPATH}/checkpoint_model.pth'
    if os.path.exists(SUBPATH) == False:
        os.makedirs(SUBPATH)
    retrained_model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, Checkpoint_model, NUM_EPOCH,SUBPATH )

    file1 = f'{SUBPATH}/{MODEL_NAME}.pth'
    if retrained_model == None:
        print("Training process UNFINISHED. A checkpoint model is saved.")
    else:
        torch.save(retrained_model.state_dict(), file1)
        print(f'The retrained model is saved at {file1}.')

    return retrained_model




#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == '__main__':
    print("this is called")
    datadir = './data/SNP_SNP_0.1'
    modelname = 'vgg_model_May_12'
    networkname = 'vgg16'
    Train_model(datadir,modelname, networkname)