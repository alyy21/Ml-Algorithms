import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("bank-info.csv")

# Eda PART --> Data exploration part 
# 1.null values
# 2.duplicates value
# 3.outiers
# 4.label encoading 

# 1. Null values 
n = df.isnull().sum().sum()
# print(n)
# df.info()




# 2. Duplicate value
# d = df.duplicated().sum()
# print(d)
df.drop_duplicates(inplace=True)
# d = df.duplicated().sum()
# print(d)




# 3. Outliers
for col in df.columns:
    if(df[col].dtype != 'object'):
        plt.boxplot(df[col])
        plt.xlabel(col)
        # plt.show()


out_col = ['age','campaign','cons.conf.idx']
for col in out_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lowerFence = Q1-1.5*IQR
    upperFence = Q3+1.5*IQR
    df = df[(df[col] >= lowerFence) & (df[col]<=upperFence)]


# 4. Label encoading 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if(df[col].dtype == 'object'):
        df[col]= le.fit_transform(df[col])
    
# print(df)



# VIF
x = df.drop('y',axis = 1)
y = df['y']
# print(x)
# print(y)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)

vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_df['Multocollinearity']=vif_values
# print(vif_df)


x.drop('nr.employed', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)

x.drop('cons.price.idx', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)


x.drop('pdays', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)

x.drop('euribor3m', axis = 1, inplace=True)
vif_df = pd.DataFrame()
vif_df ['Features']= x.columns
vif_values =[]
for i in range(len(x.columns)):
    vif = variance_inflation_factor(x.values,i)
    vif_values.append(vif)
vif_df['Multocollinearity']=vif_values
# print(vif_df)


x.drop('cons.conf.idx', axis = 1, inplace=True)
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


# Model Training 
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