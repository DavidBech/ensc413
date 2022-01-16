import numpy as np
import pandas as pd

# read the dataframe df from the csv file
df = pd.read_csv("polydata.csv")
# extract x and y values from the corresponding columns in the dataframe 
x = df.loc[:,'x'].values
y = df.loc[:,'y'].values
# now x and y contain the data values from a polynomial

# add your code below
