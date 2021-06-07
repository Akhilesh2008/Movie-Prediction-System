import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

attributes = ['Runtime', 'Aspect_Ratio', 'Content_Rating_Score', 'Genre_Musical', 'Genre_Romance', 'Genre_Sport', 'Genre_Crime', 'Genre_Documentary', 'Genre_Film-Noir', 'Genre_Short', 'Genre_Fantasy', 'Genre_Horror', 'Genre_Comedy', 'Genre_Western', 'Genre_Thriller', 'Genre_War', 'Genre_Animation', 'Genre_Family', 'Genre_Mystery', 'Genre_Adventure', 'Genre_Drama', 'Genre_History', 'Genre_Biography', 'Genre_Sci-Fi', 'Genre_News', 'Genre_Action', 'Release_Month', 'Director_Avg_Movie_Revenue', 'Lead_Actor_Avg_Movie_Revenue', 'Budget', 'Lead_Actor_Name', 'Director_Name', 'Revenue', 'class']

df = pd.read_csv('real.csv')
print(df.head())

X = np.array(df.iloc[:,0:30])
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/5,random_state=0)



k_value = np.arange(1,80,2)
# k_value = list(range(1,100, 2))
k_acc_score =[ ]

print(k_value)
for k in k_value:
    clf=RandomForestClassifier(n_estimators=k+20, random_state=42,n_jobs=-1)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    k_acc_score.append(accuracy_score(y_test,y_pred))

print(type(k_value))
# print(k_acc_score)
plt.plot(k_value, k_acc_score)
plt.xlabel("No. of trees")
plt.ylabel("Accuracy")
plt.scatter(k_value, k_acc_score)
plt.grid()
plt.show()





# clf=RandomForestClassifier(n_estimators=20, random_state=42,n_jobs=-1)

# clf.fit(X_train,y_train)

# y_pred=clf.predict(X_test)




# print("accuracy : {}".format(accuracy_score(y_test,y_pred)))




