from msilib.schema import Directory
import torch
import cv2 as cv
from PIL import Image
from torchvision import transforms
import os


model = torch.hub.load('pytorch/vision:v0.10.0','resnet18',pretrained = True) 
model.eval()

fp = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/n13054560/n13054560_10.JPEG"
input_image = Image.open(fp)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
if torch.cuda.is_available():
    print("USING GPU")
    input_batch = input_batch.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output = model(input_batch)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
top3_prob, top3_catid = torch.topk(probabilities, 10)

print(fp)
for i in range(top3_prob.size(0)):
    print(categories[top3_catid[i]], top3_prob[i].item())
