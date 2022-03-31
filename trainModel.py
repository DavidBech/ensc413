import deepLearningModel
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

def plotHistory(history):
    plt.plot(history.history['categorical_accuracy'], 'ko')
    plt.plot(history.history['val_categorical_accuracy'], 'b')
    plt.title('Accuracy vs Training Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()

if __name__ == "__main__":
    trainLoc = "./Data/train"
    testLoc = "./Data/test"
    weightSaveName = "temp"

    model = deepLearningModel.getModel()

    history = deepLearningModel.trainModel(model, trainLoc, testLoc, weightSaveName)

    plotHistory(history)


