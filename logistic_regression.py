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
    
print(df)