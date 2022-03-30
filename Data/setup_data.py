import shutil
import os
import create_data_common
import random 
import glob

# Data to include as well as the quantity of data
includeRealTrain = True
realTrainPercent = 1.00

includeUnityTrain = True
unityTrainPercent = 1.00

includeBlenderTrain = False
blenderTrainPercent = 1.00

# location of test and train out paths
testOutPath = "test"
trainOutPath = "train"

def copyFiles(fromDir, toDir, namePreppend, copyPercent):
    coppiedCount = 0
    fileCount = 0
    for category in os.listdir(fromDir):
        fromPath = fromDir + "/" + category
        toPath = toDir + "/" + category + "/"

        for file in os.listdir(fromPath):
            fileCount += 1
            if random.random() < copyPercent:
                shutil.copy(fromPath + "/" + file, toPath + "/" + namePreppend + file)
                coppiedCount += 1
    print(f"Finished Copying: {coppiedCount} out of {fileCount}")

if __name__ == "__main__":
    create_data_common.setupDirs(testOutPath, trainOutPath)

    if includeRealTrain:
        print("Starting Copying Real Training Files")
        copyFiles("./train_real", "./train", "real", realTrainPercent)

    if includeUnityTrain:
        print("Starting Copying Unity Training Files")
        copyFiles("./train_unity", "./train", "unity", unityTrainPercent)
    
    if includeBlenderTrain:
        print("Starting Copying Blender Training Files")
        copyFiles("./train_blender", "./train", "blender", blenderTrainPercent)

