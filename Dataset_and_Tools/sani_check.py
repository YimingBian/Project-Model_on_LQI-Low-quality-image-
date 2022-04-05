import shutil, os

from numpy import full
modes = ["SNP"]
sets = ["train", "val"]
things = ["frying pan", "goldfinch", "hamster", "pitcher", "upright"]
mother_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/"
dest_modes = ["SNP_small"]
for i in range(len(modes)):
    for set in sets:
        for thing in things:
            src = f"{mother_dir}{modes[i]}/{set}/{thing}"
            src_files = os.listdir(src)
            dest = f"{mother_dir}{dest_modes[i]}/{set}/{thing}"
            print(dest)
            if set == "train":
                for ii in range(100):
                    shutil.copy(os.path.join(src,src_files[ii]), dest)
            else:
                for ii in range(50):
                    shutil.copy(os.path.join(src,src_files[ii]), dest)
             
#src_files = os.listdir(src)
#dest_train_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP/train/upright"
#dest_val_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data/SNP/val/upright"
#total_num = len(src_files)
#train_num =int(total_num*2/3) 
#val_num = total_num - train_num

#for i in range(150):
#    full_name = os.path.join(src, src_files[i])
#    if i<100:
#        shutil.copy(full_name, dest_train_dir)
#    else:
#        shutil.copy(full_name, dest_val_dir)


