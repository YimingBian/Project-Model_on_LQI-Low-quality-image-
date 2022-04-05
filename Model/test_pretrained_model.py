from msilib.schema import Directory
import torch
import cv2 as cv
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn
from torchvision import datasets, models, transforms


#model = torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained = True) 
model = models.resnet18(pretrained=True)
#PATH = "./model_ft_SNP_small.pth"
#model = models.resnet18()
#for param in model.parameters():
#    param.requires_grad = False
#model.fc = nn.Linear(512,5)
#model.load_state_dict(torch.load(PATH))

model.eval()

fp = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/n03841143(odometer)/n03841143_793.JPEG"
input_image = Image.open(fp)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


datadir = 'D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test_ori'
testset = datasets.ImageFolder(datadir,transform=data_transforms)
classes = testset.classes
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
#if torch.cuda.is_available():
#    print("USING GPU")
#    input_batch = input_batch.to('cuda')
#    model.to('cuda')
for data in testloader:

    images, labels = data
    images = images.to('cuda')
    model = model.to('cuda')
    with torch.no_grad():
    #    output = model(input_batch)
        output = model(images)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)


    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    _,predict_id = torch.topk(probabilities,1)
    prediction = categories[predict_id]
    #print(f'correct: {classes[labels.item()]}')
    print(f'prediction: {prediction}')