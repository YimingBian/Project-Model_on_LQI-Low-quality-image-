import os
import cv2
import MODULE_Construction_detail as consD
srcdir_part = './Testing_data/SNP'
destdir_part = './Testing_data/MF_SNP'
SNP_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
objects = ['goldfinch', 'vizsla', 'pitcher', 'upright', 'hamster']
MF_ksize = [3,5,7,9]

for SNP_level in SNP_levels:
    for ksize in MF_ksize:
        for obj in objects:
            src_dir = f'{srcdir_part}/SNP_{SNP_level}/test/{obj}' 
            dest_dir = f'{destdir_part}/SNP_{SNP_level}/ksize_{ksize}/test/{obj}'
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                print(f'New Directory: {dest_dir}\n')
            file_lst = consD.getListOfFiles(src_dir)
            for ori_img in file_lst:
                im_tmp = cv2.imread(ori_img)
                im_out = cv2.medianBlur(im_tmp, ksize)

                prev_name = ori_img.split('.JPEG')[0].replace(src_dir,dest_dir)
                new_name = f'{prev_name}_MF.JPEG'
                cv2.imwrite(new_name, im_out)



#list_of_files = consD.getListOfFiles(srcdir)
#prev_name = list_of_files[0].split('.JPEG')[0].replace('tmp','target')
#new_name = f'{prev_name}_MF.JPEG'

#im_dir = 'data/SNP_SNP_0.3/test/goldfinch'
#im1 = cv2.imread(f'{im_dir}/n01531178_62_SNP_0.3.JPEG')
#cv2.imshow('winname', im1)
#
#im2 = cv2.medianBlur(im1,7)
#

#cv2.imwrite(new_name,)