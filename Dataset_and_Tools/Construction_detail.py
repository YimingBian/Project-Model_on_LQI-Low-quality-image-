import os
'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles    

lvl1noise = 0
lvl2noise = 0
lvl3noise = 0
lvl4noise = 0

DIR = './SNP/val'
allFiles = getListOfFiles(DIR)

for f in allFiles:
    if '_SNP_0.1' in f:
        lvl1noise += 1
    elif '_SNP_0.2' in f:
        lvl2noise += 1
    elif '_SNP_0.3' in f:
        lvl3noise += 1
    elif '_SNP_0.4' in f:
        lvl4noise += 1

all_im = lvl1noise+lvl2noise+lvl3noise+lvl4noise

print(f'# images with level 1 noise: {100*lvl1noise/all_im}%')
print(f'# images with level 2 noise: {100*lvl2noise/all_im}%')
print(f'# images with level 3 noise: {100*lvl3noise/all_im}%')
print(f'# images with level 4 noise: {100*lvl4noise/all_im}%')

