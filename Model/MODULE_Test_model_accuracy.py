import numpy as np
import torch.nn as nn
import torch
from MODULE_TNT import test_model, test_single_pretrained_model
from MODULE_Pie_graph import Draw_pie_graphs

def Modify_last_fc_layer(NETOWKRNAME, ODIM):
    """
    Usage of Modify_last_fc_layer(NETOWKRNAME, ODIM):
    *********************************************************************************
    This function modifies the last fc layer to the specified ODIM(Output dimension)
    1. NETOWKRNAME: a list of pretrained networks is here "https://pytorch.org/vision/stable/models.html"
    2. ODIM: output dimension
    """
    model_tmp = torch.hub.load('pytorch/vision:v0.10.0',NETOWKRNAME,pretrained = True)
    for params in model_tmp.parameters():
        params.requires_grad = False
    if NETOWKRNAME == 'resnet18':
        indim = model_tmp.fc.in_features
        model_tmp.fc = nn.Linear(indim, ODIM)
    elif NETOWKRNAME == 'vgg16':
        indim = model_tmp.classifier[6].in_features
        model_tmp.classifier[6] = nn.Linear(indim, ODIM, bias=True)
    return model_tmp

def Test_accuracy_retrained_models(MODELPATHs, MODELNAMEs, NETWORKNAME, ODIM, NOISETYPE, TESTSETPATHs, PIE = False, WRITE = False):
    """
    Usage of Test_accuracy_retrained_models(MODELPATHs, MODELNAMEs, NETWORKNAME, ODIM, NOISETYPE, TESTSETPATHs, PIE = False, WRITE = False):
    *********************************************************************************
    1. MODELPATHs: a list of paths of models(.pth)
    2. MODELNAMEs: a list of model names(string). e.g. MODELNAME = ["mix model", "SNP model"]
    3. NETWORKNAME: a list of pretrained networks is here "https://pytorch.org/vision/stable/models.html"
    4. ODIM:output dimensions
    5. NOISETYPE: Currently two options: 'SNP' and 'GS'. More to be added
    6. TESTSETPATHs: a list of paths of testing sets. e.g. TESTDIRS = ["D:/test_ori", "D:/test"] 
    7. PIE = False
    8. WRITE = False 
    *********************************************************************************
    """
    for testsetpath in TESTSETPATHs:
        err_distribution = list()
        for i in range(len(MODELPATHs)):
            modelname = MODELNAMEs[i]
            model = Modify_last_fc_layer(NETWORKNAME, ODIM)
            model.load_state_dict(torch.load(MODELPATHs[i]))
            print("starting passing test set ...\n")
            err, lab = test_model(model1=model, datadir=testsetpath, noisetype = NOISETYPE, writemode = WRITE, filename = modelname)
            err_distribution.append(err)
        if PIE:
            num_parts = len(lab)
            print(num_parts)
            explode = np.zeros(num_parts)
            start_angle = np.zeros(num_parts)
            leg = [True for i in range(len(MODELPATHs))]
            file_name = f'err_dist_{modelname}_{NOISETYPE}.png'
            Draw_pie_graphs(err_distribution, lab, MODELNAMEs, explode, start_angle, leg, 1, FIGSIZE=(20,20), FILENAME=file_name)

if __name__ == '__main__':
    #an example
    TESTDIRS = ["D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test_ori",
               "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test"] 
    MODELDIRS = [   "./model_ft_mix_small.pth",
                    "./model_ft_SNP_small.pth"]
    MODELNAME = ["mix model", "SNP model"]
    networkname = "resnet18"
    output_dim = 5
    noise = "SNP"
    Test_accuracy_retrained_models(MODELDIRS, MODELNAME, networkname, output_dim, noise, TESTDIRS)