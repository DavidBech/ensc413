import shutil
import os
import create_data_common
import random 
import glob
import json

# Data to include in training
includeRealTrain = True
realTrainPercent = 0.1

includeUnityTrain = False
unityTrainPercent = 1.00

includeBlenderTrain = False
blenderTrainPercent = 1.00

# Data to include in testing
includeRealTest = True
realTestPercent = 0.1

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
    categoryCount = {}

    for category in os.listdir(fromDir):
        fromPath = fromDir + "/" + category
        totalFiles += len(os.listdir(fromPath))

    for category in os.listdir(fromDir):
        fromPath = fromDir + "/" + category
        toPath = toDir + "/" + category + "/"
        categoryCount[category] = 0
        for file in os.listdir(fromPath):
            fileCount += 1
            if random.random() < copyPercent:
                shutil.copy(fromPath + "/" + file, toPath + "/" + namePreppend + file)
                coppiedCount += 1
                categoryCount[category] += 1
            if (fileCount % 1000) == 0:
                print(f"Processed {fileCount} out of {totalFiles}")
        
    print(f"Finished -- copied: {coppiedCount} out of {totalFiles}")
    return categoryCount

if __name__ == "__main__":
    create_data_common.setupDirs(testOutPath, trainOutPath)
    realTrainFiles = None
    unityTrainFiles = None
    blenderTrainFiles = None
    realTestFiles = None
    unityTestFiles = None
    blenderTestFiles = None

    if includeRealTrain:
        print("Starting Copying Real Training Files")
        realTrainFiles = copyFiles("./train_real", "./train", "real", realTrainPercent)

    if includeUnityTrain:
        print("Starting Copying Unity Training Files")
        unityTrainFiles = copyFiles("./train_unity", "./train", "unity", unityTrainPercent)
    
    if includeBlenderTrain:
        print("Starting Copying Blender Training Files")
        blenderTrainFiles = copyFiles("./train_blender", "./train", "blender", blenderTrainPercent)

    if includeRealTest:
        print("Starting Copying Real Testing Files")
        realTestFiles = copyFiles("./test_real", "./test", "real", realTestPercent)

    if includeUnityTest:
        print("Starting Copying Unity Testing Files")
        unityTestFiles = copyFiles("./test_unity", "./test", "unity", unityTestPercent)
    
    if includeBlenderTest:
        print("Starting Copying Blender Testing Files")
        blenderTestFiles = copyFiles("./test_blender", "./test", "blender", blenderTestPercent)

    jsonStr = f"""{{
    "train" : {{
        "real" : {{
            "use": {"true" if includeRealTrain else "false"},
            "percent": {realTrainPercent},
            "files": "{realTrainFiles:}"
        }},
        "unity" : {{
            "use": {"true" if includeUnityTrain else "false"},
            "percent": {unityTrainPercent},
            "files": "{unityTrainFiles}"

        }},
        "blender" : {{
            "use": {"true" if includeBlenderTrain else "false"},
            "percent": {blenderTrainPercent},
            "files": "{blenderTrainFiles}"
        }}
    }},
    "test" : {{
        "real" : {{
            "use": {"true" if includeRealTest else "false"},
            "percent": {realTestPercent},
            "files": "{realTestFiles}"
        }},
        "unity" : {{
            "use": {"true" if includeUnityTest else "false"},
            "percent": {unityTestPercent},
            "files": "{unityTestFiles}"

        }},
        "blender" : {{
            "use": {"true" if includeBlenderTest else "false"},
            "percent": {blenderTestPercent},
            "files": "{blenderTestFiles}"
        }}   
    }}  
}}
""" 
    with open("./data_setup.json", "w") as f:
        print(jsonStr, file=f)


