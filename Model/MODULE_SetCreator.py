import shutil, os
import random
import glob

def Create_train_val_data(src_mother_dir, src_folder, dest_mother_dir, dest_folder, set_name, num_per_cat, ratio = 2/3, WindowsOS = False, Pattern = False):
    for i in range(len(src_folder)):
        src = os.path.join(src_mother_dir,src_folder[i],set_name)
        if Pattern == False:
            src_files = os.listdir(src)
            dest_train_dir = os.path.join(dest_mother_dir,set_name,"train",dest_folder[i])
            dest_val_dir = os.path.join(dest_mother_dir,set_name,"val",dest_folder[i])
        else:
            src_files = glob.glob(f'{src}/{Pattern}')
            for ii in range(len(src_files)):
                if WindowsOS:
                    src_files[ii] = src_files[ii].replace(f'{src}\\','')
                else:
                    src_files[ii] = src_files[ii].replace(f'{src}/','')
            pt = Pattern.replace('*','')
            dest_train_dir = os.path.join(dest_mother_dir,f'{set_name}_{pt}',"train",dest_folder[i])
            dest_val_dir = os.path.join(dest_mother_dir,f'{set_name}_{pt}',"val",dest_folder[i])

        if WindowsOS:
            src = src.replace('/','\\')
            dest_train_dir = dest_train_dir.replace('/','\\')
            dest_val_dir = dest_val_dir.replace('/','\\')

        if not os.path.exists(dest_train_dir):
            os.makedirs(dest_train_dir)
            print(f'folder created: {os.path.join("train",dest_folder[i])} \n')
        if not os.path.exists(dest_val_dir):
            os.makedirs(dest_val_dir)
            print(f'folder created: {os.path.join("val",dest_folder[i])} \n')

        train_num = int(num_per_cat*ratio)

        rand_idx = random.sample(range(0, len(src_files)), num_per_cat)

        for i in range(num_per_cat):
            full_name = os.path.join(src, src_files[rand_idx[i]])
            if i<train_num:
                shutil.copy(full_name, dest_train_dir)
            else:
                shutil.copy(full_name, dest_val_dir)     

def Create_test_data(src_mother_dir, src_folder, dest_mother_dir, dest_folder, set_name , num_per_cate, mode = "test", WindowsOS = False, Pattern = False):
    for i in range(len(src_folder)):
        src = os.path.join(src_mother_dir,src_folder[i],set_name)
        if Pattern == False:
            src_files = os.listdir(src)
            dest_test_dir = os.path.join(dest_mother_dir,set_name,mode,dest_folder[i])
        else:
            src_files = glob.glob(f'{src}/{Pattern}')
            for ii in range(len(src_files)):
                if WindowsOS:
                    src_files[i] = src_files[ii].replace(f'{src}\\','')
                else:
                    src_files[i] = src_files[ii].replace(f'{src}/','')
            pt = Pattern.replace('*','')
            dest_test_dir = os.path.join(dest_mother_dir,f'{set_name}_{pt}',mode,dest_folder[i])
            

        if WindowsOS:
            src = src.replace('/','\\')
            dest_test_dir = dest_test_dir.replace('/','\\')

        if not os.path.exists(dest_test_dir):
            os.makedirs(dest_test_dir)
            print(f'folder created: {os.path.join(mode,dest_folder[i])} \n')

        rand_idx = random.sample(range(0, len(src_files)), num_per_cate)

        for i in range(num_per_cate):
            full_name = os.path.join(src, src_files[rand_idx[i]])
            shutil.copy(full_name, dest_test_dir)

# an example
#src_mother_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator"
#source_folders = ["n01531178(goldfinch)","n02342885(hamster)","n03400231(frying pan)","n03950228(pitcher)","n04515003(upright)"]
#
#dest_mother_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data"
#dest_folders = ["goldfinch","hamster","frying pan","pitcher","upright"]
#
#total_num = 300
#
##Create_train_val_data(  src_mother_dir=src_mother_dir, src_folder=source_folders, dest_mother_dir=dest_mother_dir,
##                        dest_folder=dest_folders, set_name="GS", total_num=total_num, WindowsOS=True)
#
#Create_test_data(   src_mother_dir=src_mother_dir, src_folder=source_folders, dest_mother_dir=dest_mother_dir,
#                    dest_folder=dest_folders, set_name="GS", num_per_cate=20, mode="test_small", WindowsOS=True)