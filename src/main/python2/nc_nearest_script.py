import os
from extra_functions import list_image_dir

if __name__ == '__main__':
    settingsDir = os.path.join('..', '..', '..', 'config')
    settingsFileName = os.path.join(settingsDir, 'nc.conf')

    rootDir = ''
    baseName = ''
    netFileName = ''
    maxPatchLevelQuery = 0
    maxPatchLevelRef = 0
    descrVecLen = 0
    GPU_MODE = 0

    with open(settingsFileName, 'r') as settingsFile:
        settings = settingsFileName.readlines()
        for param in settings:
            exec param

    workDir = os.path.join(rootDir, 'collections', baseName)
    imageBasePath = os.path.join(workDir, 'images')
    imageList = list_image_dir(imageBasePath)
    ncPath = os.path.join(workDir, 'neuralcodes')
    netsPath = os.path.join(rootDir, 'nets')
    netFullName = os.path.join(netsPath, netFileName)
