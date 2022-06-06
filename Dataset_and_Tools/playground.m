folders = ["./n01531178(goldfinch)","./n03400231(frying pan)","./n02342885(hamster)","./n03950228(pitcher)","./n04515003(upright)"];

for f = 1:length(folders)
    files = dir(fullfile(folders(f),'*.JPEG'));
    modes = ["SNP", "GS"];

    for i = 1:length(files)
        base_file_name = files(i).name;
        full_file_name = fullfile(files(i).folder, base_file_name);
        for j = 1:length(modes)
            PIC_generator(full_file_name,modes(j))
        end
    end
end
disp("done")
