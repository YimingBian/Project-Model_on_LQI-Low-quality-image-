import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import torch
from torchvision import datasets, transforms

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def train_model(model, criterion, optimizer, scheduler, DATALOADER, DEVICE, DATASETSIZE, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataloaders = DATALOADER
    device = DEVICE 
    dataset_sizes = DATASETSIZE
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

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

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, DATALOADER, DEVICE, CLASSNAMES, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    dataloaders = DATALOADER
    device = DEVICE
    class_names = CLASSNAMES

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def test_model(model, datadir, batchsize):
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    testset = datasets.ImageFolder(datadir,transform=data_transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

    correct = 0
    total = 0
    count = 0
    with torch.no_grad():
        for data in testloader:
            count += 1
            images, labels = data
            #print(labels)
            outputs = model(images)
            print(outputs)
            _,predicted = torch.max(outputs.data,1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

    print(f'Accuracy of the network on {total} test images: {100 * correct // total} %')

def compare_models(model1, model2, datadir):
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    testset = datasets.ImageFolder(datadir,transform=data_transforms)
    classes = testset.classes
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    correct1 = 0
    total = 0
    correct2 = 0
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs1 = model1(images)
            outputs2 = model2(images)
            total += labels.size(0)
            #for retrained model
            _,predicted1 = torch.max(outputs1.data,1)
            correct1 += (predicted1 == labels).sum().item()

            # for original model
            with open("imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            
            probabilities = torch.nn.functional.softmax(outputs2[0], dim=0)
            _,predict_id = torch.topk(probabilities,1)
            prediction = categories[predict_id]

            if prediction == classes[labels.item()]:
                correct2 += 1

    print(f'Accuracy of the retrained network on {total} test images: {100 * correct1 // total} %')
    print(f'Accuracy of the original network on {total} test images: {100 * correct2 // total} %')

def test_single_model(model1, datadir, writemode = False, filename = None):
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    testset = datasets.ImageFolder(datadir,transform=data_transforms)
    classes = testset.classes
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    correct1 = 0
    total = 0
    model1.eval()

    model1=model1.to('cuda')
    
    count = 0
    lvl_1_err = 0
    lvl_2_err = 0
    lvl_3_err = 0
    lvl_4_err = 0
    ori_err = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            total += labels.size(0)
        
            images = images.to('cuda')
        
            outputs1 = model1(images)
            #for retrained model
            _,predicted1 = torch.max(outputs1.data,1)
            
            predicted1=predicted1.to('cpu')
            
            correct1 += (predicted1 == labels).sum().item()
            if predicted1 != labels:
                wrong_filename = str(testloader.dataset.imgs[count][0]).split('\\')[-1]
                if '_SNP_0.4.JPEG' in wrong_filename:
                    lvl_4_err += 1
                elif '_SNP_0.3.JPEG' in wrong_filename:
                    lvl_3_err += 1
                elif '_SNP_0.2.JPEG' in wrong_filename:
                    lvl_2_err += 1
                elif '_SNP_0.1.JPEG' in wrong_filename:
                    lvl_1_err += 1
                else:
                    ori_err += 1
            count += 1

    if writemode == False:
        print(f'Accuracy of the retrained network on {total} test images: {100 * correct1 / total} %')
        print(f'original: {ori_err} ({100*ori_err/total}%)')
        print(f'lvl1: {lvl_1_err} ({100*lvl_1_err/total}%)')
        print(f'lvl2: {lvl_2_err} ({100*lvl_2_err/total}%)')
        print(f'lvl3: {lvl_3_err} ({100*lvl_3_err/total}%)')
        print(f'lvl4: {lvl_4_err} ({100*lvl_4_err/total}%)')
    #elif 'ori' in datadir or 'ORI' in datadir:
    #    txtdir = "./Results/"+filename+'.txt'
    #    F = open(txtdir,'a')
    #    F.write(f'\nAccuracy of the retrained network on {total} test images: {100 * correct1 / total} %\n')
    #    F.write('==========\n')
    #    F.close
    else:
        txtdir = "./Results/"+filename+'.txt'
        F = open(txtdir,'a')
        F.write(f'\nAccuracy of the retrained network on {total} test images: {100 * correct1 / total} %\n')
        F.write(f'original: {ori_err} ({100*ori_err/total}%)\n')
        F.write(f'lvl1: {lvl_1_err} ({100*lvl_1_err/total}%)\n')
        F.write(f'lvl2: {lvl_2_err} ({100*lvl_2_err/total}%)\n')
        F.write(f'lvl3: {lvl_3_err} ({100*lvl_3_err/total}%)\n')
        F.write(f'lvl4: {lvl_4_err} ({100*lvl_4_err/total}%)\n')
        F.write('==========\n')
        F.close


def test_single_pretrained_model(model1, datadir, writemode = False , filename = None):
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    testset = datasets.ImageFolder(datadir,transform=data_transforms)
    classes = testset.classes
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    correct1 = 0
    total = 0
    model1.eval()

    model1=model1.to('cuda')
    
    count = 0
    lvl_1_err = 0
    lvl_2_err = 0
    lvl_3_err = 0
    lvl_4_err = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images=images.to('cuda')

            outputs1 = model1(images)
            total += labels.size(0)

            with open("imagenet_classes.txt", "r") as f:
                categories = [s.strip() for s in f.readlines()]
            
            probabilities = torch.nn.functional.softmax(outputs1[0], dim=0)
            _,predict_id = torch.topk(probabilities,1)
            
            predict_id=predict_id.to('cpu')
            
            prediction = categories[predict_id]

            if prediction == classes[labels.item()]:
                correct1 += 1
            else:
                wrong_filename = str(testloader.dataset.imgs[count][0]).split('\\')[-1]
                if '_SNP_0.4.JPEG' in wrong_filename:
                    lvl_4_err += 1
                elif '_SNP_0.3.JPEG' in wrong_filename:
                    lvl_3_err += 1
                elif '_SNP_0.2.JPEG' in wrong_filename:
                    lvl_2_err += 1
                elif '_SNP_0.1.JPEG' in wrong_filename:
                    lvl_1_err += 1
                else:
                    ori_err += 1
            count += 1

    if writemode == False:
        print(f'Accuracy of the retrained network on {total} test images: {100 * correct1 / total} %')
        print(f'original: {ori_err} ({100*ori_err/total}%)')
        print(f'lvl1: {lvl_1_err} ({100*lvl_1_err/total}%)')
        print(f'lvl2: {lvl_2_err} ({100*lvl_2_err/total}%)')
        print(f'lvl3: {lvl_3_err} ({100*lvl_3_err/total}%)')
        print(f'lvl4: {lvl_4_err} ({100*lvl_4_err/total}%)')
    #elif 'ori' in datadir or 'ORI' in datadir:
    #    txtdir = "./Results/"+filename+'.txt'
    #    F = open(txtdir,'a')
    #    F.write(f'\nAccuracy of the retrained network on {total} test images: {100 * correct1 / total} %\n')
    #    F.write('==========\n')
    #    F.close
    else:
        txtdir = "./Results/"+filename+'.txt'
        F = open(txtdir,'a')
        F.write(f'\nAccuracy of the retrained network on {total} test images: {100 * correct1 / total} %\n')
        F.write(f'original: {ori_err} ({100*ori_err/total}%)\n')
        F.write(f'lvl1: {lvl_1_err} ({100*lvl_1_err/total}%)\n')
        F.write(f'lvl2: {lvl_2_err} ({100*lvl_2_err/total}%)\n')
        F.write(f'lvl3: {lvl_3_err} ({100*lvl_3_err/total}%)\n')
        F.write(f'lvl4: {lvl_4_err} ({100*lvl_4_err/total}%)\n')
        F.write('==========\n')
        F.close