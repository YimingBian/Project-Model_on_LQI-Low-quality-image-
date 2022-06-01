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

def get_only_file_name(dirName):
    list_files = os.listdir(dirName)
    return list_files

def Get_dataset_construction(directory, keys, keywords, pct = False):
    allFiles = getListOfFiles(directory)
    construction = {}
    for i in range(len(keys)):
        counter = 0
        for f in allFiles:
            if keywords[i] in f:
                counter = counter + 1
        construction[keys[i]] = counter
    if pct:
        pct_list = list()
        tt = sum(construction.values())
        for i in range(len(construction)):
            pct_list.append(round(construction[keys[i]]/tt*100,2))
            pass
        return construction,pct_list
    else:
        return construction

if __name__ == "__main__":
    # an example
    keys = ["pythonfiles", "matlabfiles"]
    keywords = ['.py', '.m']
    DIR = './'

    A,B = Get_dataset_construction(DIR,keys,keywords,True)
    #A = Get_dataset_construction(DIR,keys,keywords)

    print(A)
    print(B)