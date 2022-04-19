from MODULE_Test_model_accuracy import Test_accuracy_retrained_models

TESTDIRS = ["D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS/test"]
MODELDIRS = [   "./model_ft_ori_small.pth",
                "./model_ft_SNP_small.pth",
                "./model_ft_mix_small.pth"]
MODELNAME = ["ori_model_GS", "SNP_model_GS", "mix_model_GS"]
networkname = "resnet18"
output_dim = 5
noise = "GS"
Test_accuracy_retrained_models(MODELDIRS, MODELNAME, networkname, output_dim, noise, TESTDIRS, PIE=True, WRITE=True)