import MODULE_Construction_detail as consD
import os 
list_file = consD.getListOfFiles('./Testing_data/SNP/SNP_0.1')
for file in list_file:
    if '_MF' in file:
        print(f'{file} found & removed!')
        os.remove(file)
print('done.')
