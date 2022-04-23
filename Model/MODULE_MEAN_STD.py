import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for D in dataloader:
        data,_ = D
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean.tolist(), std.tolist()


def Mean_and_std_of_dataset(DATASETDIR, BATCHSIZE = 64, RESIZE = (224,224)):
    data_transforms = transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.ToTensor()])
    test_dataset = datasets.ImageFolder(DATASETDIR,transform=data_transforms)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCHSIZE)
    data_mean, data_std = get_mean_and_std(test_dataloader)
    return data_mean, data_std

# an example
#dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS/test"
#dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.2_v_0.2/test"
#a,b = Mean_and_std_of_dataset(dir)
#print(a)
#print(b)