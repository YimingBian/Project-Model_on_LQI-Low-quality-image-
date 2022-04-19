import numpy as np
import torch.nn as nn
import torch
from MODULE_TNT import test_single_model, test_single_pretrained_model
from MODULE_Pie_graph import Draw_pie_graphs


def Modify_last_fc_layer(NETOWKRNAME, ODIM):
    model_tmp = torch.hub.load('pytorch/vision:v0.10.0',NETOWKRNAME,pretrained = True)
    for params in model_tmp.parameters():
        params.requires_grad = False
    indim = model_tmp.fc.in_features
    model_tmp.fc = nn.Linear(indim, ODIM)
    return model_tmp

def Test_accuracy_retrained_models(MODELPATHs, MODELNAMEs, NETWORKNAME, ODIM, NOISETYPE, TESTSETPATHs, PIE = False, WRITE = False):
    for testsetpath in TESTSETPATHs:
        err_distribution = list()
        for i in range(len(MODELPATHs)):
            modelname = MODELNAMEs[i]
            model = Modify_last_fc_layer(NETWORKNAME, ODIM)
            model.load_state_dict(torch.load(MODELPATHs[i]))
            print("starting passing test set ...\n")
            err, lab = test_single_model(model1=model, datadir=testsetpath, noisetype = NOISETYPE, writemode = WRITE, filename = modelname)
            err_distribution.append(err)
        if PIE:
            num_parts = len(lab)
            print(num_parts)
            explode = np.zeros(num_parts)
            start_angle = np.zeros(num_parts)
            leg = [True for i in range(len(MODELPATHs))]
            file_name = f'err_dist_{modelname}_{NOISETYPE}.png'
            Draw_pie_graphs(err_distribution, lab, MODELNAMEs, explode, start_angle, leg, 1, FIGSIZE=(20,20), FILENAME=file_name)


# an example
#TESTDIRS = ["D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test_ori",
#           "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_small/test"] 
#MODELDIRS = [   "./model_ft_mix_small.pth",
#                "./model_ft_SNP_small.pth"]
#MODELNAME = ["mix model", "SNP model"]
#networkname = "resnet18"
#output_dim = 5
#noise = "SNP"
#Test_accuracy_retrained_models(MODELDIRS, MODELNAME, networkname, output_dim, noise, TESTDIRS)