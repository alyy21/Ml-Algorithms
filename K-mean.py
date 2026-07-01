import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


df = pd.read_csv('ODI data.csv',encoding = 'latin')
# df.info()
df[['start','end']] = df['Span'].str.split('-',n = 1,expand= True)
# print(df)
# x = df.dtypes
# print(x)


# converting all object column to numerical column 
cols = ['Span', 'Inns', 'NO','Runs', 'HS', 'Ave', 'BF','SR', '100', '50', '0','start','end']

df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# x = df.dtypes
# print(x)

# Try to make exp column using start and end

df['Exp'] = df['end']-df['start']

# print(df)

# now drop the columns span ,start and end 
df.drop(columns=['Span', 'start', 'end'], inplace = True, axis =1)
# print(df)



df.drop(columns=['Unnamed: 13'], inplace=True)
numeric_cols = df.select_dtypes(include='number').columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
# x = df.isnull().sum()
# print(x)

# y= df.duplicated().sum()
# print(y)


df_copy = df.copy()
df_copy.drop(['Player'],axis=1,inplace=True)
# print(df_copy)


# Model Training 
from sklearn.preprocessing import StandardScaler
se = StandardScaler()
df_scaled = se.fit_transform(df_copy)
# print(df_scaled)

df_scaled = pd.DataFrame(df_scaled, columns=df_copy.columns)
# print(df_scaled) 


# how to choose the numbers of clusters

# However choosing the right value of K can be tricky?

# If you choose too few clusters than you might miss some important patterns in the data
# If you choose too many clusters than the model might become too complex.

# Smaller datasets: for small dataset, you might try k-values from 1 to a value 10 or 15
# Larger dataset: for larger dataset you can explore a wider range


from sklearn.cluster import KMeans
k_values = [2,3,4,5,6,7,8]

# Elbow 
ssd = []

for k in k_values:
  km = KMeans(n_clusters=k,max_iter=150,random_state=32)
  km.fit(df_scaled)
  ssd.append(km.inertia_)

plt.plot(k_values,ssd)
# plt.show()

from sklearn.metrics import silhouette_score

k_values = range(2,8)

silhouette_scores = []
for k in k_values:
  kmeans = KMeans(n_clusters=k,random_state=32)
  kmeans.fit(df_scaled)
  silhouette_avg = silhouette_score(df_scaled,kmeans.labels_)
  silhouette_scores.append(silhouette_avg)
plt.subplot(1,2,2)
plt.plot(k_values,silhouette_scores,marker='o',color='green')
# plt.show()

# 6 will be considered as number of clusters

kmodel = KMeans(n_clusters=6, max_iter=150, random_state=32)
kmodel.fit(df_scaled)
# print(kmodel.labels_)
df['Cluster_ID'] = kmodel.labels_

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Runs', y='Ave', hue='Cluster_ID', palette='Set1',s = 150)
plt.title('This plot will show how cricketers are clustered based on runs and average')
plt.show()