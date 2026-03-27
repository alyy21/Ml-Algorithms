import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# Eda part(data cleaning , exploration)
df = pd.read_csv("heart.csv")
# print(df)
# df.info()
x = df.isnull().sum().sum()
# print(x)
y = df.duplicated().sum()
# print(y)
df.drop_duplicates(inplace=True)
y = df.duplicated().sum()
# print(y)



for col in df.columns:
    sns.boxplot(df[col])
    plt.title(col)
    # plt.show()


# MODEL BUILDING 
# split the data 
# model importation 
# train the model 
# evaluate the model 
# check the accuracy 
# hyperparameter tuning 


x = df.iloc[:,:-1]
y = df['target']
# print(x)
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80,random_state=0)
# print(x_train)
# print(y_train)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test) #predicted values
# y_test actual value 

from sklearn.metrics import *
ac = accuracy_score(y_pred, y_test)*100
# print(ac)


depth = [1,2,3,4,5,6,7,8,9,10]
for max_d in depth:
    model = DecisionTreeClassifier(max_depth=max_d, random_state=1)
    model.fit(x_train,y_train)
    y_pred= model.predict(x_test)
    ac= accuracy_score(y_pred, y_test)*100
    print("The Accuracy of the model for max depth", max_d, "is", ac)


final_model = DecisionTreeClassifier(max_depth=4,random_state=2)
final_model.fit(x_train,y_train)
y_pred= final_model.predict(x_test)
ac= accuracy_score(y_test,y_pred)*100
print(ac)
