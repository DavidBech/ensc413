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
numberOfRuns = 10000

trainingError = [[None] * (maxDeg + 1)] * (numberOfRuns) 
validateError = [[None] * (maxDeg + 1)] * (numberOfRuns) 

# 4. run steps 1-3 10k times
for run in range(numberOfRuns):
    # 1. split data into training and validation sets randomly 
    xTrain, xValidate, yTrain, yValidate = train_test_split(x, y, test_size=trainingSplit, random_state=run)
    for k in range(maxDeg + 1):
        # 2. find the coefficients of the plynomial using the training set for each of the degrees
        # fit the values for a degree k poly
        coef = np.polyfit(xTrain, yTrain, k) 

        # 3. calculate the error with respect to the training set and with respect to the validation set
        # calculate error in polynomial
        XTrain = np.vander(xTrain, k + 1)
        XVal = np.vander(xValidate, k + 1)
        # E = || y - Xa ||^2
        trainingError[run][k] = np.linalg.norm((yTrain - np.matmul(XTrain, coef)))
        validateError[run][k] = np.linalg.norm((yValidate - np.matmul(XVal, coef)))

# 5. Calculate the average error
averageTrainingError = [0] * (maxDeg + 1)
averageValidateError = [0] * (maxDeg + 1)
for i in range (numberOfRuns):
    for j in range(maxDeg + 1):
        averageTrainingError[j] += trainingError[i][j]
        averageValidateError[j] += validateError[i][j]
for j in range(maxDeg +1):
    averageTrainingError[j] /= numberOfRuns
    averageValidateError[j] /= numberOfRuns

# 6. Print results so the smallest error can be found
print(f"Average Error in Polynomials:")
for k in range(maxDeg +1):
    print(f"poly {k: <2}: Train:{averageTrainingError[k]: <20} Validate:{averageValidateError[k]}")

# For Verification of polynomials matching data plot the polynomials using all data
# Outupt polynomials graphically using all data 
polyList = [None] * (maxDeg + 1)
plotList = [None] * (maxDeg + 1)
for k in range(maxDeg + 1):
    # fit the values for a degree k poly
    coef = np.polyfit(x, y, k) 
    
    # store the polynomial
    polyList[k] = np.poly1d(coef)
    plotList[k] = plt.plot(x, polyList[k](x), label = f"poly-degree:{k}")

# plot all the results
raw_data = plt.scatter(x, y, c = "c", marker = "x", label = "Input Data")
plt.legend(handles=plotList.append(raw_data))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
quit()
