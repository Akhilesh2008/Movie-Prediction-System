import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#attributes = ['sepal_length','sepal_width','petal_length','petal_width','class']

df =pd.read_csv('real.csv')
#print(df.head())
X= np.array(df.iloc[:,0:10])
y= np.array(df['class'])

print(X)
print(y)
