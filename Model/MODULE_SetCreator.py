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
            # to be deleted
            print(f'dest_train_dir:{dest_train_dir}')
            print(f'dest_val_dir:{dest_val_dir}')
            #            
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

# default proportions: [train, val, test] = [80%, 10%, 10%]
def Create_all_sets(src_mother_dir, src_folders, dest_mother_dir1, dest_mother_dir2, 
            dest_folders, noise_type , num_per_cat, Pattern = False):
    """
    src_mother_dir = "./Whole_data"
    dest_mother_dir1 = "./Training_data"
    dest_mother_dir2 = "./Testing_data"
    """
    for i in range(len(src_folders)):
#        src = os.path.join(src_mother_dir,noise_type,src_folders[i])
        dst = []
        if Pattern == False:
            src = os.path.join(src_mother_dir,src_folders[i])
            # need to be modified
            src_files = os.listdir(src)
            for ii in range(len(src_files)):
                src_files[ii] = src_files[ii].replace(f'{src}/','')

            dest_train_dir = os.path.join(dest_mother_dir1,noise_type,"train",dest_folders[i])
            dest_val_dir = os.path.join(dest_mother_dir1,noise_type,"val",dest_folders[i])
            dest_test_dir = os.path.join(dest_mother_dir2, noise_type, "test",dest_folders[i])

            dst.append(dest_train_dir)
            dst.append(dest_val_dir)
            dst.append(dest_test_dir)

        else:
            src = os.path.join(src_mother_dir,noise_type,src_folders[i])
            src_files = glob.glob(f'{src}/{Pattern}')
            for ii in range(len(src_files)):
                src_files[ii] = src_files[ii].replace(f'{src}/','')
            ###
            print(f'There are {len(src_files)} images matching the pattern.')
            # ###

            pt = Pattern.replace('*','')
            dest_train_dir = os.path.join(dest_mother_dir1,noise_type,f'{noise_type}_{pt}',"train",dest_folders[i])
            dest_train_whole_dir = os.path.join(dest_mother_dir1,noise_type,f'{noise_type}_whole',"train",dest_folders[i])
            dest_val_dir = os.path.join(dest_mother_dir1,noise_type,f'{noise_type}_{pt}',"val",dest_folders[i])
            dest_val_whole_dir = os.path.join(dest_mother_dir1,noise_type,f'{noise_type}_whole',"val",dest_folders[i])
            dest_test_dir = os.path.join(dest_mother_dir2, noise_type, f'{noise_type}_{pt}/test',dest_folders[i])
            dest_test_whole_dir = os.path.join(dest_mother_dir2, noise_type, f'{noise_type}_whole/test',dest_folders[i])

            dst.append(dest_train_dir)
            dst.append(dest_train_whole_dir)
            dst.append(dest_val_dir)
            dst.append(dest_val_whole_dir)
            dst.append(dest_test_dir)
            dst.append(dest_test_whole_dir)
            

        for dir in dst:
            if not os.path.exists(dir):
                os.makedirs(dir)
                print(f'folder created: {dir} \n')
        
        start_idx_train = 0
        end_idx_train = int(num_per_cat * 0.8)
        start_idx_val = end_idx_train
        end_idx_val = start_idx_val + int(num_per_cat * 0.1)
        start_idx_test = end_idx_val 
        end_idx_test = num_per_cat

        ###
        print(f'train range {start_idx_train}, {end_idx_train}')
        print(f'val range {start_idx_val}, {end_idx_val}')
        print(f'test range {start_idx_test}, {end_idx_test}')
        
        ###

        for train_idx in range(start_idx_train, end_idx_train):
            full_name = os.path.join(src, src_files[train_idx])
            shutil.copy(full_name, dest_train_dir)
            if not Pattern == False: 
                shutil.copy(full_name, dest_train_whole_dir)
            
        
        for val_idx in range(start_idx_val, end_idx_val):
            full_name = os.path.join(src, src_files[val_idx])
            shutil.copy(full_name, dest_val_dir)
            if not Pattern == False:
                shutil.copy(full_name, dest_val_whole_dir)
        
        for test_idx in range(start_idx_test, end_idx_test):
            full_name = os.path.join(src, src_files[test_idx])
            shutil.copy(full_name, dest_test_dir)
            if not Pattern == False:
                shutil.copy(full_name, dest_test_whole_dir)
                    
        
        
if __name__ == '__main__':
    #an example
    #src_mother_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator"
    #source_folders = ["n01531178(goldfinch)","n02342885(hamster)","n03400231(frying pan)","n03950228(pitcher)","n04515003(upright)"]
    #
    #dest_mother_dir = "D:/Academic/2022Spring/575/Project/Model/PIC_generator/data"
    #dest_folders = ["goldfinch","hamster","frying pan","pitcher","upright"]
    #
    #total_num = 300
    #
    #Create_train_val_data(  src_mother_dir=src_mother_dir, src_folder=source_folders, dest_mother_dir=dest_mother_dir,
    #                        dest_folder=dest_folders, set_name="GS", num_per_cat=total_num, WindowsOS=True)
    #
    #Create_test_data(   src_mother_dir=src_mother_dir, src_folder=source_folders, dest_mother_dir=dest_mother_dir,
    #                    dest_folder=dest_folders, set_name="GS", num_per_cate=20, mode="test_small", WindowsOS=True)
    src_mother_dir = "./Whole_data"
    cat_folders = ['vizsla','upright','pitcher','hamster','goldfinch']
    dest_mother_dir1 = "./Training_data"
    dest_mother_dir2 = "./Testing_data"
    noise_type = 'SNP'
    num_per_cat = 10
    pattern = '*0.1*'

    Create_all_sets(src_mother_dir=src_mother_dir, src_folders=cat_folders, dest_mother_dir1=dest_mother_dir1,
                    dest_mother_dir2=dest_mother_dir2, dest_folders=cat_folders, noise_type=noise_type, num_per_cat=num_per_cat,
                    Pattern=pattern)
    
     