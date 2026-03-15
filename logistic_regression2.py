import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("heart.csv")

# x = df.isnull().sum().sum()
# print(x)

# y = df.duplicated().sum()
# print(y)
# df.info()
df.drop_duplicates(inplace=True)
d = df.duplicated().sum()
# print(d)

# Outliers
for col in df.columns:
    if(df[col].dtype != 'object'):
        plt.boxplot(df[col])
        plt.xlabel(col)
        # plt.show()

out_col = ['trestbps','chol','thalach','oldpeak','ca','thal']
for col in out_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lowerFence = Q1-1.5*IQR
    upperFence = Q3+1.5*IQR
    df = df[(df[col] >= lowerFence) & (df[col]<=upperFence)]


# print(df.columns)

# Split features and target
x = df.drop('target', axis=1)   # change if needed
y = df['target']

# print(x.head())
# print(y.head())

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)

vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_df['Multocollinearity']=vif_values
# print(vif_df)

x.drop('trestbps', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)

x.drop('thalach', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)


x.drop('age', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)


x.drop('chol', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)

x.drop('thal', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)




# model training 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
# print(y_pred)
# print(y_test)

from sklearn.metrics import *
ac = accuracy_score(y_test,y_pred)*100
cf=confusion_matrix(y_test,y_pred)
print(cf)
print(ac)
sns.heatmap(cf, annot=True, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()