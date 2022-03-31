from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import deepLearningModel


def generateClassifcationReport(model, test_gen):
    target_names = ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR', 'Empty', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR']
    test_gen.reset()

    # get the model's outputs
    Y_pred = model.predict(test_gen)
    classes = test_gen.classes[test_gen.index_array]
    y_pred = np.argmax(Y_pred, axis= -1)
    print(sum(y_pred==classes)/800)

    # TODO -- calculate Empty category separately

    # produce a confusion matrix
    data = confusion_matrix(classes, y_pred)
    df_cm = pd.DataFrame(data, columns=target_names, index = target_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (20,14))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size
    print('Confusion Matrix')
    print(data)
    print('Classification Report')
    print(classification_report(test_gen.classes[test_gen.index_array], y_pred, target_names=target_names))


if __name__ == "__main__":
    modelWeightsFileName = "model_temp_weights.h5"
    testLocation = "./Data/test"
    image_size = (224, 224)
    batch_size = 32

    model = deepLearningModel.getModel()

    model.summary()

    model.load_weights(modelWeightsFileName)

    test_gen = deepLearningModel.getTestGen(testLocation)

    generateClassifcationReport(model, test_gen)