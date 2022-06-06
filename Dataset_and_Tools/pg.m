folders = [".\n03400231(frying pan)"];
for f = 1:length(folders)
    files = dir(fullfile(folders(f),'*.JPEG'));
    mode = "SNP";
    
    for i = 1:length(files)
        base_file_name = files(i).name;
        full_file_name = fullfile(files(i).folder, base_file_name);
        PIC_generator(full_file_name,mode)
    end
end