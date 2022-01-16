import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read the dataframe df from the csv file
df = pd.read_csv("polydata.csv")
# extract x and y values from the corresponding columns in the dataframe 
x = df.loc[:,'x'].values
y = df.loc[:,'y'].values
# now x and y contain the data values from a polynomial

# add your code below

maxDeg = 10
trainingSplit = 0.4
randomState = 32
xTrain, xValidate, yTrain, yValidate = train_test_split(x, y, test_size=trainingSplit, random_state=randomState)

#degree k goes from 0 to 10
polyList = [None] * (maxDeg + 1)
plotList = [None] * (maxDeg + 1)
trainingError = [None] * (maxDeg + 1)
validateError = [None] * (maxDeg + 1)
for k in range(maxDeg + 1):
    # fit the values for a degree k poly
    coef = np.polyfit(xTrain, yTrain, k) 

    # store the polynomial
    polyList[k] = np.poly1d(coef)

    # calculate error in polynomial
    XTrain = np.vander(xTrain, k + 1)
    XVal = np.vander(xValidate, k + 1)
    # E = || y - Xa ||^2
    trainingError[k] = np.linalg.norm((yTrain - np.matmul(XTrain, coef)))
    validateError[k] = np.linalg.norm((yValidate - np.matmul(XVal, coef)))
    print(f"Error In poly {k: <2}: Train:{trainingError[k]: <20} Validate:{validateError[k]}")

    # plot polynomial if error is sufficiently small (value from successive runs)
    if validateError[k] < 100:
        plotList[k] = plt.plot(x, polyList[k](x), label = f"poly-degree:{k}")


# plot all the results
raw_data = plt.scatter(x, y, c = "c", marker = "x", label = "Input Data")
plt.legend(handles=plotList.append(raw_data))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
quit()
