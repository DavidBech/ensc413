import deepLearningModel
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    trainLoc = "./Data/train"
    testLoc = "./Data/test"
    runName = "temp"

    model = deepLearningModel.getModel()

    deepLearningModel.trainModel(model, trainLoc, testLoc, runName)

