function PIC_generator(imgdir, mode)
%SNP_GENERATOR Summary of this function goes here
%   Detailed explanation goes here
ori_img = imread(imgdir);

%disp(ori_img)

%dir_dic = split(imgdir,'/');
dir_dic = split(imgdir,'\');

ori_img_name = string(dir_dic(end));
ori_img_name_noext = strrep(ori_img_name,".JPEG",'');

disp(ori_img_name);
disp(ori_img_name_noext);


if (mode == "SNP") % Salt and Pepper
    d_val = [0.1 0.2 0.3 0.4];
    for i = 1:length(d_val)
        tmp_img = sprintf('im%d',i);
        tmp_img = imnoise(ori_img,"salt & pepper", d_val(i));
        
        append_name = sprintf("_SNP_%0.1f.JPEG",d_val(i));
        filename=strcat(ori_img_name_noext,append_name);
        %new_dir = strrep(imgdir,ori_img_name,strcat("SNP/",filename));
        new_dir = strrep(imgdir,ori_img_name,strcat("SNP\",filename));
        
        
        %disp(append_name);
        %disp(filename);
        %disp(new_dir);
        
        imwrite(tmp_img,new_dir);
    end
elseif (mode == "QTCP") % Quarter Crop
    r = 448;
    c = 448;
    Resize_img = imresize(ori_img, [512 512]);
    crop_window = centerCropWindow2d([512 512], [r c]);
    Center_crop_img = imcrop(Resize_img, crop_window);

    append_name = "_UL.JPEG"; % Upper Left
    filename = strcat(ori_img_name_noext,append_name);
    new_dir = strrep(imgdir,ori_img_name,strcat("QTCP\",filename));
    imwrite(Center_crop_img(1:r/2,1:c/2,:), new_dir);

    append_name = "_UR.JPEG"; % Upper Right
    filename = strcat(ori_img_name_noext,append_name);
    new_dir = strrep(imgdir,ori_img_name,strcat("QTCP\",filename));
    imwrite(Center_crop_img(1:r/2,c/2+1:c,:), new_dir);

    append_name = "_LL.JPEG"; % Lower Left
    filename = strcat(ori_img_name_noext,append_name);
    new_dir = strrep(imgdir,ori_img_name,strcat("QTCP\",filename));
    imwrite(Center_crop_img(r/2+1:r,1:c/2,:), new_dir);

    append_name = "_LR.JPEG"; % Lower Right
    filename = strcat(ori_img_name_noext,append_name);
    new_dir = strrep(imgdir,ori_img_name,strcat("QTCP\",filename));
    imwrite(Center_crop_img(r/2+1:r,c/2+1:c,:), new_dir);
    
else
    disp("Please enter a correct mode")
end

