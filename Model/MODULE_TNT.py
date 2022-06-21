import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import torch
from torchvision import datasets, transforms
from MODULE_MEAN_STD import Mean_and_std_of_dataset


def train_model(model, criterion, optimizer, scheduler, DATALOADER, DEVICE, DATASETSIZE, checkpoint_model = None, num_epochs = 25, store_path = None):
    """
    This function trains the model. Input parameters are:
    1. model: pre-trained model, or a model with random initialization 
    2. criterion: loss function, e.g. "nn.CrossEntropyLoss()"
    3. optimizer: optimizer, e.g. "optim.SGD(model.parameters(), lr=0.001, momentum=0.9)"
    4. scheduler: slowly decrease the learning rate, e.g. "lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)"
    5. DATALOADER: as suggested by name
    6. DEVIDE: as suggested by name
    7. DATASETSIZE: total number of images, not the individual category
    8. num_epochs: # of training epochs, 25 by default
    9. store_path: not needed for Nova
    """
    #f = open(f'{store_path}/Training_Epoch.txt', 'w')
    since = time.time()
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes = DATASETSIZE
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        #f.write(f'Epoch {epoch}/{num_epochs - 1}\n')
        #f.write('----------\n')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in DATALOADER[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

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
            #f.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
            

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
            
            torch.save( {
                        'Break epoch' : epoch,
                        'model_state_dict' : best_model_weights,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss
                        }, checkpoint_model)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    #f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
    #f.write(f'Best val Acc: {best_acc:4f}\n')
    

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

def test_model(model1, datadir, noisetype = 'SNP', detailmode = False, writemode = False, filename = None):
    """
    This function test the accuracy of the model.
    Parameters:
    1. model1: model to be tested
    2. datadir: the path of the folder that contains the images. e.g. ".../test"
    3. noisetype: 'SNP' by default
    4. writemode: 'False' by default. It specifies whether to store the results in a txt file. For Nova, it is False.
    5. filename: 'None' by default. If writemode is True, it needs to be specified. It is the name of the text file without extension.   
    """
    ds_mean, ds_std = Mean_and_std_of_dataset(datadir, 1)
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(ds_mean, ds_std)])

    testset = datasets.ImageFolder(datadir,transform=data_transforms)
    classes = testset.classes
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    correct1 = 0
    total = 0
    err_dist = list()
    label_dic = list()
    model1.eval()
    model1=model1.to('cuda')

    if detailmode == True:  # for testing on comprehensive testing sets
        if noisetype == 'SNP':
            # noise type is set default as salt and pepper
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

            if writemode == False:  # print on screen
                print(f'Accuracy of the retrained network on {total} test images: {100 * correct1 / total} %')
                print(f'original: {ori_err} ({100*ori_err/total}%)')
                print(f'lvl1: {lvl_1_err} ({100*lvl_1_err/total}%)')
                print(f'lvl2: {lvl_2_err} ({100*lvl_2_err/total}%)')
                print(f'lvl3: {lvl_3_err} ({100*lvl_3_err/total}%)')
                print(f'lvl4: {lvl_4_err} ({100*lvl_4_err/total}%)')
            else:                   # save in file
#                txtdir = "./Results/"+filename+'.txt'
                rdir = datadir.replace('test','test_result')
                txtdir = f'{rdir}/{filename}.txt'

                print(f'The result is stored at {txtdir}\n')
                with open(txtdir, 'a') as F:
                    F.write(f'\nAccuracy of the retrained network on {total} test images: {100 * correct1 / total} %\n')
                    F.write(f'original: {ori_err} ({100*ori_err/total}%)\n')
                    F.write(f'lvl1: {lvl_1_err} ({100*lvl_1_err/total}%)\n')
                    F.write(f'lvl2: {lvl_2_err} ({100*lvl_2_err/total}%)\n')
                    F.write(f'lvl3: {lvl_3_err} ({100*lvl_3_err/total}%)\n')
                    F.write(f'lvl4: {lvl_4_err} ({100*lvl_4_err/total}%)\n')
                    F.write('==========\n')
                F.close
            # for analysis
            err_dist.append(correct1)        
            err_dist.append(ori_err)        
            err_dist.append(lvl_1_err)
            err_dist.append(lvl_2_err)
            err_dist.append(lvl_3_err)
            err_dist.append(lvl_4_err)
            label_dic.append("Correct")
            label_dic.append("ori_err")
            label_dic.append("lvl_1_err")
            label_dic.append("lvl_2_err")
            label_dic.append("lvl_3_err")
            label_dic.append("lvl_4_err")
            return err_dist, label_dic

        elif noisetype == 'GS':
            count = 0
            m2v2_err = 0
            m2v3_err = 0
            m2v4_err = 0
            m2v5_err = 0
            m3v2_err = 0
            m3v3_err = 0
            m3v4_err = 0
            m3v5_err = 0
            m4v2_err = 0
            m4v3_err = 0
            m4v4_err = 0
            m4v5_err = 0
            m5v2_err = 0
            m5v3_err = 0
            m5v4_err = 0
            m5v5_err = 0

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
                        if '_GS_m_0.2_v_0.2.JPEG' in wrong_filename:
                            m2v2_err += 1
                        elif '_GS_m_0.2_v_0.3.JPEG' in wrong_filename:
                            m2v3_err += 1
                        elif '_GS_m_0.2_v_0.4.JPEG' in wrong_filename:
                            m2v4_err += 1
                        elif '_GS_m_0.2_v_0.5.JPEG' in wrong_filename:
                            m2v5_err += 1

                        elif '_GS_m_0.3_v_0.2.JPEG' in wrong_filename:
                            m3v2_err += 1
                        elif '_GS_m_0.3_v_0.3.JPEG' in wrong_filename:
                            m3v3_err += 1
                        elif '_GS_m_0.3_v_0.4.JPEG' in wrong_filename:
                            m3v4_err += 1
                        elif '_GS_m_0.3_v_0.5.JPEG' in wrong_filename:
                            m3v5_err += 1

                        elif '_GS_m_0.4_v_0.2.JPEG' in wrong_filename:
                            m4v2_err += 1
                        elif '_GS_m_0.4_v_0.3.JPEG' in wrong_filename:
                            m4v3_err += 1
                        elif '_GS_m_0.4_v_0.4.JPEG' in wrong_filename:
                            m4v4_err += 1
                        elif '_GS_m_0.4_v_0.5.JPEG' in wrong_filename:
                            m4v5_err += 1

                        elif '_GS_m_0.5_v_0.2.JPEG' in wrong_filename:
                            m5v2_err += 1
                        elif '_GS_m_0.5_v_0.3.JPEG' in wrong_filename:
                            m5v3_err += 1
                        elif '_GS_m_0.5_v_0.4.JPEG' in wrong_filename:
                            m5v4_err += 1
                        elif '_GS_m_0.5_v_0.5.JPEG' in wrong_filename:
                            m5v5_err += 1

                        else:
                            ori_err += 1
                    count += 1

            if writemode == False:  # print on screen
                print(f'Accuracy of the retrained network on {total} test images: {100 * correct1 / total} %')
                print(f'original: {ori_err} ({100*ori_err/total}%)')
                print(f'm_0.2_v_0.2: {m2v2_err} ({100*m2v2_err/total}%)')
                print(f'm_0.2_v_0.3: {m2v3_err} ({100*m2v3_err/total}%)')
                print(f'm_0.2_v_0.4: {m2v4_err} ({100*m2v4_err/total}%)')
                print(f'm_0.2_v_0.5: {m2v5_err} ({100*m2v5_err/total}%)')

                print(f'm_0.3_v_0.2: {m3v2_err} ({100*m3v2_err/total}%)')
                print(f'm_0.3_v_0.3: {m3v3_err} ({100*m3v3_err/total}%)')
                print(f'm_0.3_v_0.4: {m3v4_err} ({100*m3v4_err/total}%)')
                print(f'm_0.3_v_0.5: {m3v5_err} ({100*m3v5_err/total}%)')

                print(f'm_0.4_v_0.2: {m4v2_err} ({100*m4v2_err/total}%)')
                print(f'm_0.4_v_0.3: {m4v3_err} ({100*m4v3_err/total}%)')
                print(f'm_0.4_v_0.4: {m4v4_err} ({100*m4v4_err/total}%)')
                print(f'm_0.4_v_0.5: {m4v5_err} ({100*m4v5_err/total}%)')

                print(f'm_0.5_v_0.2: {m5v2_err} ({100*m5v2_err/total}%)')
                print(f'm_0.5_v_0.3: {m5v3_err} ({100*m5v3_err/total}%)')
                print(f'm_0.5_v_0.4: {m5v4_err} ({100*m5v4_err/total}%)')
                print(f'm_0.5_v_0.5: {m5v5_err} ({100*m5v5_err/total}%)')

            else:                   # save in file
#                txtdir = "./Results/"+filename+'.txt'
                rdir = datadir.replace('test','test_result')
                txtdir = f'{rdir}/{filename}.txt'
                print(f'The result is stored at {txtdir}\n')

                with open(txtdir, 'a') as F:
                    F.write(f'\nAccuracy of the retrained network on {total} test images: {100 * correct1 / total} %\n')
                    F.write(f'original: {ori_err} ({100*ori_err/total}%)\n')

                    F.write(f'm_0.2_v_0.2: {m2v2_err} ({100*m2v2_err/total}%)\n')
                    F.write(f'm_0.2_v_0.3: {m2v3_err} ({100*m2v3_err/total}%)\n')
                    F.write(f'm_0.2_v_0.4: {m2v4_err} ({100*m2v4_err/total}%)\n')
                    F.write(f'm_0.2_v_0.5: {m2v5_err} ({100*m2v5_err/total}%)\n')

                    F.write(f'm_0.3_v_0.2: {m3v2_err} ({100*m3v2_err/total}%)\n')
                    F.write(f'm_0.3_v_0.3: {m3v3_err} ({100*m3v3_err/total}%)\n')
                    F.write(f'm_0.3_v_0.4: {m3v4_err} ({100*m3v4_err/total}%)\n')
                    F.write(f'm_0.3_v_0.5: {m3v5_err} ({100*m3v5_err/total}%)\n')

                    F.write(f'm_0.4_v_0.2: {m4v2_err} ({100*m4v2_err/total}%)\n')
                    F.write(f'm_0.4_v_0.3: {m4v3_err} ({100*m4v3_err/total}%)\n')
                    F.write(f'm_0.4_v_0.4: {m4v4_err} ({100*m4v4_err/total}%)\n')
                    F.write(f'm_0.4_v_0.5: {m4v5_err} ({100*m4v5_err/total}%)\n')

                    F.write(f'm_0.5_v_0.2: {m5v2_err} ({100*m5v2_err/total}%)\n')
                    F.write(f'm_0.5_v_0.3: {m5v3_err} ({100*m5v3_err/total}%)\n')
                    F.write(f'm_0.5_v_0.4: {m5v4_err} ({100*m5v4_err/total}%)\n')
                    F.write(f'm_0.5_v_0.5: {m5v5_err} ({100*m5v5_err/total}%)\n')

                    F.write('==========\n')
                F.close

            err_dist.append(correct1)        
            err_dist.append(ori_err)    
            err_dist.append(m2v2_err)
            err_dist.append(m2v3_err)
            err_dist.append(m2v4_err)
            err_dist.append(m2v5_err)

            err_dist.append(m3v2_err)
            err_dist.append(m3v3_err)
            err_dist.append(m3v4_err)
            err_dist.append(m3v5_err)

            err_dist.append(m4v2_err)
            err_dist.append(m4v3_err)
            err_dist.append(m4v4_err)
            err_dist.append(m4v5_err)

            err_dist.append(m5v2_err)
            err_dist.append(m5v3_err)
            err_dist.append(m5v4_err)
            err_dist.append(m5v5_err)

            label_dic.append("Correct")
            label_dic.append("ori_err")
            label_dic.append("m2v2_err")
            label_dic.append("m2v3_err")
            label_dic.append("m2v4_err")
            label_dic.append("m2v5_err")

            label_dic.append("m3v2_err")
            label_dic.append("m3v3_err")
            label_dic.append("m3v4_err")
            label_dic.append("m3v5_err")

            label_dic.append("m4v2_err")
            label_dic.append("m4v3_err")
            label_dic.append("m4v4_err")
            label_dic.append("m4v5_err")

            label_dic.append("m5v2_err")
            label_dic.append("m5v3_err")
            label_dic.append("m5v4_err")
            label_dic.append("m5v5_err")

    else: # on single noise testing sets
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                total += labels.size(0)
                images = images.to('cuda')
                outputs1 = model1(images)
                #for retrained model
                _,predicted1 = torch.max(outputs1.data,1)
                predicted1 = predicted1.to('cpu')
                correct1 += (predicted1 == labels).sum().item()
        if writemode == False:  # print on screen
            acc = round(100 * correct1 / total, 2)
            print(f'Accuracy of the retrained network on {total} test images: {acc} %')
            result_dir = f'./Results/Re-trained_models/{filename}/{filename}.txt'
            with open(result_dir, 'a') as F:
                F.write(f'{acc}\n')
            F.close


        else:                   # save in file
#               txtdir = "./Results/"+filename+'.txt'
            rdir = datadir.replace('test','test_result')
            txtdir = f'{rdir}/{filename}.txt'
            print(f'The testing result of {model1} is stored at {txtdir}\n')
            with open(txtdir, 'a') as F:
                F.write(f'\nAccuracy of the retrained network on {total} test images: {100 * correct1 / total} %\n')
                F.write('==========\n')
            F.close
        # for analysis
    err_dist.append(correct1)        
    err_dist.append(total-correct1)        
    label_dic.append("Correct")
    label_dic.append("error")
    return err_dist, label_dic

 


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

def test_model_USELESS(model, datadir, batchsize):
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

def test_single_pretrained_model(model1, datadir, writemode = False , filename = None):
    ds_mean, ds_std = Mean_and_std_of_dataset(datadir, 1)
    data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(ds_mean, ds_std)])
    testset = datasets.ImageFolder(datadir,transform=data_transforms)
    classes = testset.classes
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    correct1 = 0
    total = 0
    model1.eval()

    model1=model1.to('cuda')
    
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


    if writemode == False:
        print(f'Dataset: {datadir}\n')
        print(f'Accuracy on {total} test images: {100 * correct1 / total} %\n')
        print('*'*10)

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
        F.write('==========\n')
        F.close

if __name__ == "__main__":
    pass
