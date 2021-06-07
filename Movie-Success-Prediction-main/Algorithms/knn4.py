import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

attributes = ['Runtime', 'Aspect_Ratio', 'Content_Rating_Score', 'Genre_Musical', 'Genre_Romance', 'Genre_Sport', 'Genre_Crime', 'Genre_Documentary', 'Genre_Film-Noir', 'Genre_Short', 'Genre_Fantasy', 'Genre_Horror', 'Genre_Comedy', 'Genre_Western', 'Genre_Thriller', 'Genre_War', 'Genre_Animation', 'Genre_Family', 'Genre_Mystery', 'Genre_Adventure', 'Genre_Drama', 'Genre_History', 'Genre_Biography', 'Genre_Sci-Fi', 'Genre_News', 'Genre_Action', 'Release_Month', 'Director_Avg_Movie_Revenue', 'Lead_Actor_Avg_Movie_Revenue', 'Budget', 'Lead_Actor_Name', 'Director_Name', 'Revenue', 'class']
df =pd.read_csv('real.csv')
# print(df.head())
# 176
i=0
b = 'Ben Affleck'
a = np.array(df['Lead_Actor_Name'])
c = np.array(df['Lead_Actor_Avg_Movie_Revenue'])
for x in a:
    i=i+1
    if(x==b):
        avg_actor = int(c[i-1])
        break
print(avg_actor)
print(i)
# X= np.array(df.iloc[:,30:31])
# y= np.array(df['class'])
# print(a[0])
# print(X)
# print(y)

