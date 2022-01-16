import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the dataframe df from the csv file
df = pd.read_csv("polydata.csv")
# extract x and y values from the corresponding columns in the dataframe 
x = df.loc[:,'x'].values
y = df.loc[:,'y'].values
# now x and y contain the data values from a polynomial

# add your code below

maxDeg = 10

#degree k goes from 0 to 10
polyList = [None] * (maxDeg + 1)
plotList = [None] * (maxDeg + 1)
polyError = [None] * (maxDeg + 1)
for k in range(maxDeg + 1):
    # fit the values for a degree k poly
    coef = np.polyfit(x, y, k) 

    # store the polynomial
    polyList[k] = np.poly1d(coef)

    # calculate error in polynomial
    X = np.vander(x, k + 1)
    # E = || y - Xa ||^2
    polyError[k] = np.linalg.norm((y - np.matmul(X, coef)))
    print(f"Error In poly {k: <2}: {polyError[k]}")

    # plot polynomial if error is sufficiently small (value from successive runs)
    if polyError[k] < 139:
        plotList[k] = plt.plot(x, polyList[k](x), label = f"poly-degree:{k}")


# plot all the results
raw_data = plt.scatter(x, y, c = "c", marker = "x", label = "Input Data")
plt.legend(handles=plotList.append(raw_data))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
quit()
