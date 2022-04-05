import shutil, os

from numpy import full
src = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/n04515003(upright)/SNP"
src_files = os.listdir(src)
dest_train_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP/train/upright"
dest_val_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP/val/upright"
total_num = len(src_files)
train_num =int(total_num*2/3) 
val_num = total_num - train_num

for i in range(total_num):
    full_name = os.path.join(src, src_files[i])
    if i<train_num:
        shutil.copy(full_name, dest_train_dir)
    else:
        shutil.copy(full_name, dest_val_dir)
