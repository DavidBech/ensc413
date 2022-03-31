import shutil
import os
import create_data_common
import random 
import glob

# Data to include in training
includeRealTrain = True
realTrainPercent = 1.00

includeUnityTrain = False
unityTrainPercent = 1.00

includeBlenderTrain = False
blenderTrainPercent = 1.00

# Data to include in testing
includeRealTest = True
realTestPercent = 1.00

includeUnityTest = False
unityTestPercent = 1.00

includeBlenderTest = False
blenderTestPercent = 1.00

# location of test and train out paths
testOutPath = "test"
trainOutPath = "train"

def copyFiles(fromDir, toDir, namePreppend, copyPercent):
    coppiedCount = 0
    fileCount = 0
    totalFiles = 0

    for category in os.listdir(fromDir):
        fromPath = fromDir + "/" + category
        totalFiles += len(os.listdir(fromPath))

    for category in os.listdir(fromDir):
        fromPath = fromDir + "/" + category
        toPath = toDir + "/" + category + "/"

        for file in os.listdir(fromPath):
            fileCount += 1
            if random.random() < copyPercent:
                shutil.copy(fromPath + "/" + file, toPath + "/" + namePreppend + file)
                coppiedCount += 1
            if (fileCount % 1000) == 0:
                print(f"Processed {fileCount} out of {totalFiles}")
        
    print(f"Finished -- copied: {coppiedCount} out of {totalFiles}")

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

    if includeRealTest:
        print("Starting Copying Real Testing Files")
        copyFiles("./test_real", "./test", "real", realTestPercent)

    if includeUnityTest:
        print("Starting Copying Unity Testing Files")
        copyFiles("./test_unity", "./test", "unity", unityTestPercent)
    
    if includeBlenderTest:
        print("Starting Copying Blender Testing Files")
        copyFiles("./test_blender", "./test", "blender", blenderTestPercent)

