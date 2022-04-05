folders = [".\n03400231(frying pan)"];
for f = 1:length(folders)
    files = dir(fullfile(folders(f),'*.JPEG'));
    %modes = ["SNP", "QTCP"];
    mode = "QTCP";
    
    for i = 1:length(files)
        base_file_name = files(i).name;
        full_file_name = fullfile(files(i).folder, base_file_name);
        full_file_name_win = strrep(full_file_name,'/','\');
        %for j = 1:length(modes)
        %    PIC_generator(full_file_name_win,modes(j))
        %end
        PIC_generator(full_file_name_win,mode)
    end
end