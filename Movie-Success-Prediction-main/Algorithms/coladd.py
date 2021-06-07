import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

attributes = ['Runtime', 'Aspect_Ratio', 'Content_Rating_Score', 'Genre_Musical', 'Genre_Romance', 'Genre_Sport', 'Genre_Crime', 'Genre_Documentary', 'Genre_Film-Noir', 'Genre_Short', 'Genre_Fantasy', 'Genre_Horror', 'Genre_Comedy', 'Genre_Western', 'Genre_Thriller', 'Genre_War', 'Genre_Animation', 'Genre_Family', 'Genre_Mystery', 'Genre_Adventure', 'Genre_Drama', 'Genre_History', 'Genre_Biography', 'Genre_Sci-Fi', 'Genre_News', 'Genre_Action', 'Release_Month', 'Director_Avg_Movie_Revenue', 'Lead_Actor_Avg_Movie_Revenue', 'Budget', 'Lead_Actor_Name', 'Director_Name', 'Revenue', 'class']

df =pd.read_csv('real2.csv')
print(df.head())

X= np.array(df.iloc[:,0:4])
y= np.array(df['class'])
z= np.array(df['Lead_Actor_Avg_Movie_Revenue'])
k=np.array(df['Lead_Actor_Avg_Movie_Revenue'])
i=0
for a in y:
    i=i+1
    if(a==2):
        # print("hello")
        # print(i)
        # print(type(a))
        k[i-1]=str(z[i-1]+50)
    elif(a==3):
        k[i-1]=str(z[i-1]+100)
    elif(a==4):
        k[i-1]=str(z[i-1]+150)
    elif(a==5):
        k[i-1]=str(z[i-1]+200)
    else:
        k[i-1]=str(z[i-1])
        

print(k)
allcc = pd.Series(k)


# df.drop(columns=['Lead_Actor_Avg_Movie_Revenue'])
# list(map(int, ['1','2','3']))

df['Lead_Actor_Avg_Movie_Revenue']=allcc

# print(type(df['Lead_Actor_Avg_Movie_Revenue']))
# Lead_Actor_Avg_Movie_Revenue=k
# print(type(allcc))
# print(X)
# print(y)

ya= np.array(df['class'])
za= np.array(df['Director_Avg_Movie_Revenue'])
ka=np.array(df['Director_Avg_Movie_Revenue'])
ia=0
for aa in ya:
    ia=ia+1
    if(aa==2):
        # print("hello")
        # print(i)
        # print(type(a))
        ka[ia-1]=str(za[ia-1]+50)
    elif(aa==3):
        ka[ia-1]=str(za[ia-1]+100)
    elif(aa==4):
        ka[ia-1]=str(za[ia-1]+150)
    elif(aa==5):
        ka[ia-1]=str(za[ia-1]+200)
    else:
        ka[ia-1]=str(za[ia-1])
        

print(ka)
allcca = pd.Series(ka)

df['Director_Avg_Movie_Revenue']=allcca


yaa= np.array(df['class'])
zaa= np.array(df['Budget'])
kaa=np.array(df['Budget'])
iaa=0
for aaa in yaa:
    iaa=iaa+1
    if(aaa==2):
        # print("hello")
        # print(i)
        # print(type(a))
        kaa[iaa-1]=str(abs(zaa[iaa-1]-20))
    elif(aaa==3):
        kaa[iaa-1]=str(abs(zaa[iaa-1]-50))
    elif(aaa==4):
        kaa[iaa-1]=str(abs(zaa[iaa-1]-70))
    elif(aaa==5):
        kaa[iaa-1]=str(abs(zaa[iaa-1]-100))
    else:
        kaa[iaa-1]=str(abs(zaa[iaa-1]))
        

print(kaa)
allccaa = pd.Series(kaa)

df['Budget']=allccaa



err=[1,2,3,9] #1
drr=[3,7,8,9] #2
crr=[4,6,7] #3
brr=[5,6,10] #4
arr=[5,11] #5


yaz= np.array(df['class'])
zaz= np.array(df['Release_Month'])
kaz=np.array(df['Release_Month'])
iaz=0
for aaz in yaz:
    iaz=iaz+1
    if(aaz==2):
        n = random.randint(0,3)
        kaz[iaz-1]=str(drr[n])
    elif(aaz==3):
        n = random.randint(0,2)
        kaz[iaz-1]=str(crr[n])
    elif(aaz==4):
        n = random.randint(0,2)
        kaz[iaz-1]=str(brr[n])
    elif(aaz==5):
        n = random.randint(0,1)
        kaz[iaz-1]=str(arr[n])
    else:
        n = random.randint(0,3)
        kaz[iaz-1]=str(err[n])
        

print(kaz)
allccaz = pd.Series(kaz)

df['Release_Month']=allccaz

# err=[1,2,3,7,8,9] #1
# drr=[3,7,8,9,10] #2
# crr=[4,5,6,11] #3
# brr=[5,6,11] #4
# arr=[5,11] #5

# randomlist = []
# for i in range(0,10):
# n = random.randint(0,5)
# print(arr[n])
    # randomlist.append(n)
    # print(randomlist)


df.to_csv("new.csv")





# df["Lead_Actor_Avg_Movie_Revenue"]=k
# print(df)

# d=np.array(df['Lead_Actor_Avg_Movie_Revenue'])
# print(d)
# df["Lead_Actor_Avg_Movie_Revenue"]=k
# X= np.array(df.iloc[:,0:35])

# print(X[1])

# z= np.array(df.iloc[:,0:6])
# print(z)

