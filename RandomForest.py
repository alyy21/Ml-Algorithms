import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv("customer_churn.csv")
# df.info()


# Eda PART

# n = df.isnull().sum().sum()
# print(n)
# d = df.duplicated().sum()
# print(d)



# Label Encoading
df=df.drop('customerID',axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if (df[col].dtype == 'object')  &  (col !="Churn") :
        df[col]=le.fit_transform(df[col])

# print(df)
# df.info()

# Model Training 
x = df.drop('Churn', axis = 1)
y = df['Churn']

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split( x, y , test_size=0.1 , random_state=0)


# model building
from sklearn.ensemble import RandomForestClassifier

model_1 = RandomForestClassifier(n_estimators=200) #n_estimators= To define how many decision tree you want to use inside your random forest model

# Train this model 1 by using training data (x_train , y_train)
model_1.fit(x_train , y_train)

# Ask the model to predict churn value for testing data (x_test)
y_pred = model_1.predict(x_test)

# Now use original testing data churn values  (y_test ) to compare predicted churn values
from sklearn.metrics import accuracy_score
# print("Accuracy score of this Random Forest model is :", accuracy_score(y_test , y_pred)*100)




# check which n estimator value is suitable 
n_estimators_data = [5,10,25,50,100,150,200,250,300,500,1000]
for i in n_estimators_data:
  model_1 = RandomForestClassifier(n_estimators = i)  # n_estimators= To define how many decision tree you want to use inside your random forest model
  # Train this model 1 by using training data (x_train , y_train)
  model_1.fit(x_train , y_train)
  # Ask the model to predict churn value for testing data (x_test)
  y_pred = model_1.predict(x_test)
  # Now use original testing data churn values  (y_test ) to compare predicted churn values
  from sklearn.metrics import accuracy_score
  print("Accuracy score of this Random Forest model with DT no:" ,  i  , "is:", accuracy_score(y_test , y_pred)*100)


# Hyper parameter tunning 
# Grid Search CV
# Simply create a dict of all hyper parameters that you want to try
paramaters_data = { "random_state" : [0,42,60,100], "n_estimators" : [5,25,50,100,200,300,500],
                  "max_depth" :[5,10,15,20], "criterion" : ["gini","entropy"] }

# Now build a random forest model that will be trained again and again by using diff diff of hyperparamters from the grid
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()

# Apply gridsearch cv on this model to try parameters
grid_search = GridSearchCV( estimator = model , param_grid = paramaters_data   , scoring="accuracy" )
grid_search.fit(x_train , y_train)

print("Best Paramaters for this random forest model is :", grid_search.best_params_)




# Build a final Random Forest model with best Hyperparaters selected by Grid Search cv library 
model_1 = RandomForestClassifier(criterion ='gini' , max_depth = 10, n_estimators = 200, random_state =42)
# Train this model 1 by using training data (x_train , y_train)
model_1.fit(x_train , y_train)
# Ask the model to predict churn value for testing data (x_test)
y_pred = model_1.predict(x_test)
# Now use original testing data churn values  (y_test ) to compare predicted churn values
from sklearn.metrics import accuracy_score
print("Accuracy score of this Random Forest model with Best hyperparaters is:", accuracy_score(y_test , y_pred)*100)