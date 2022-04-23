from MODULE_Test_model_accuracy import Test_accuracy_retrained_models
from MODULE_SetCreator import Create_train_val_data, Create_test_data


# an example
#src_mother_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator"
#source_folders = ["n01531178(goldfinch)","n02342885(hamster)","n03400231(frying pan)","n03950228(pitcher)","n04515003(upright)"]
#
#dest_mother_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data"
#dest_folders = ["goldfinch","hamster","frying pan","pitcher","upright"]
#
#num_per_cat = 300
#
#
#
#pat_dict_GS = [    "*m_0.2_v_0.2*", "*m_0.2_v_0.3*", "*m_0.2_v_0.4*", "*m_0.2_v_0.5*", 
#                "*m_0.3_v_0.2*", "*m_0.3_v_0.3*", "*m_0.3_v_0.4*", "*m_0.3_v_0.5*", 
#                "*m_0.4_v_0.2*", "*m_0.4_v_0.3*", "*m_0.4_v_0.4*", "*m_0.4_v_0.5*", 
#                "*m_0.5_v_0.2*", "*m_0.5_v_0.3*", "*m_0.5_v_0.4*", "*m_0.5_v_0.5*"]
#
#pat_dict_SNP = [    "*SNP_0.1*", "*SNP_0.2*", "*SNP_0.3*", "*SNP_0.4*"]
#
#for i in pat_dict_GS:
#    Create_train_val_data(  src_mother_dir=src_mother_dir, src_folder=source_folders, dest_mother_dir=dest_mother_dir,
#                            dest_folder=dest_folders, set_name="GS", num_per_cat=num_per_cat, WindowsOS=True, Pattern=i)
#    Create_test_data(  src_mother_dir=src_mother_dir, src_folder=source_folders, dest_mother_dir=dest_mother_dir,
#                            dest_folder=dest_folders, set_name="GS", num_per_cate=num_per_cat, WindowsOS=True, Pattern=i)
#for i in pat_dict_SNP:
#    Create_train_val_data(  src_mother_dir=src_mother_dir, src_folder=source_folders, dest_mother_dir=dest_mother_dir,
#                            dest_folder=dest_folders, set_name="SNP", num_per_cat=num_per_cat, WindowsOS=True, Pattern=i)
#    Create_test_data(  src_mother_dir=src_mother_dir, src_folder=source_folders, dest_mother_dir=dest_mother_dir,
#                            dest_folder=dest_folders, set_name="SNP", num_per_cate=num_per_cat, WindowsOS=True, Pattern=i)
MODELDIRS = [   "./model_ft_ori_small.pth",
                "./model_ft_SNP_small.pth",
                "./model_ft_mix_small.pth"]
MODELNAME = ["RtO", "RtN", "RtM"]
networkname = "resnet18"
output_dim = 5

TESTDIRS_GS = [ "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.2_v_0.2/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.2_v_0.3/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.2_v_0.4/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.2_v_0.5/test",

                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.3_v_0.2/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.3_v_0.3/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.3_v_0.4/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.3_v_0.5/test",

                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.4_v_0.2/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.4_v_0.3/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.4_v_0.4/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.4_v_0.5/test",

                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.5_v_0.2/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.5_v_0.3/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.5_v_0.4/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/GS_m_0.5_v_0.5/test"
]
          
TESTDIRS_SNP = [ "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_SNP_0.1/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_SNP_0.2/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_SNP_0.3/test",
                "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP_SNP_0.4/test"      ]


noise1 = "GS"
Test_accuracy_retrained_models(MODELDIRS, MODELNAME, networkname, output_dim, noise1, TESTDIRS_GS, PIE=False, WRITE=True)
    
noise2 = "SNP"    
Test_accuracy_retrained_models(MODELDIRS, MODELNAME, networkname, output_dim, noise2, TESTDIRS_SNP, PIE=False, WRITE=True)




#noise = "GS"
#
#Test_accuracy_retrained_models(MODELDIRS, MODELNAME, networkname, output_dim, noise, TESTDIRS, PIE=True, WRITE=True)