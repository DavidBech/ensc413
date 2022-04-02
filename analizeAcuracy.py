from contextlib import redirect_stdout
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import deepLearningModel
import json

def plotHistory(history, location):
    catAc = history["categorical_accuracy"]
    valCatAc = history["val_categorical_accuracy"]
    catAc_list = []
    valCatAc_list = []
    for key in catAc:
        catAc_list.append(catAc[key])
    for key in valCatAc:
        valCatAc_list.append(valCatAc[key])

    plt.figure(figsize=(10,10))
    plt.plot(catAc_list, 'ko')
    plt.plot(valCatAc_list, 'b')
    plt.title('Accuracy vs Training Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.savefig(location + "historyPlt.png")
    #plt.show()

def generateClassifcationReport(model, test_gen, location):
    target_names = ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR', 'Empty', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR']
    test_gen.reset()

    # get the model's outputs
    Y_pred = model.predict(test_gen)
    classes = test_gen.classes[test_gen.index_array]
    y_pred = np.argmax(Y_pred, axis= -1)
    print(sum(y_pred==classes)/800)

    # produce a confusion matrix
    data = confusion_matrix(classes, y_pred)
    
    # normalize data
    normalizedData = []
    for i in data:
        totalPoints = 0
        for j in i:
            totalPoints += j
        normalizedData.append(i/totalPoints)

    df_cm = pd.DataFrame(data, columns=target_names, index = target_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    plt.savefig(location + "confustion_matrix.png")

    df_cm = pd.DataFrame(normalizedData, columns=target_names, index = target_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    plt.savefig(location + "confustion_matrix_normalized.png")

    with open(location + "accuracyMeasures.dat", "w") as f:
        with redirect_stdout(f):
            print("labels")
            print(target_names)
            print('Confusion Matrix')
            print(data)
            print('Confusion Matrix Normalized')
            print(normalizedData)
            print('Classification Report')
            print(classification_report(test_gen.classes[test_gen.index_array], y_pred, target_names=target_names))



if __name__ == "__main__":
    runName = "unityTrain_realTest"
    dataLocation = "./RunData/" + runName + "/"
    testLocation = "./Data/test"
    image_size = (224, 224)
    batch_size = 32

    historyFileName = dataLocation + "model_history.dat"
    modelWeightFileName = dataLocation + "model_weights.h5"
    
    model = deepLearningModel.getModel()

    model.load_weights(modelWeightFileName)

    test_gen = deepLearningModel.getTestGen(testLocation)

    #generateClassifcationReport(model, test_gen, dataLocation)

    histData = None
    with open(historyFileName, "r") as history:
        histData = json.load(history)
    plotHistory(histData, dataLocation)
