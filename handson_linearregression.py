import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
insurance = pd.read_csv("insurance.csv")

# EDA(DATA EXPLORING PART)
# insurance.head(10)
# insurance.tail(10)
# insurance.info()

# Data Cleaning
'''
1. Check null values 
2. Check outliers
3. Check duplicates
'''

# 1. NUll values
# x = insurance.isnull().sum()
# y = insurance.isnull().sum().sum() # sum of null values like how many null values we have
# print(y)

'''
for numerical value -- replace the value with mean or median 
for object value -- replace the value with mode 
'''

col = list(insurance.columns)
# print(col)

for i in col:
    if insurance[i].dtype=='object':
        insurance[i]=insurance[i].fillna(insurance[i].mode()[0])
    else:
        insurance[i]=insurance[i].fillna(insurance[i].mean())

x = insurance.isnull().sum().sum()
# print(x)


#2. check duplicates  
# x = insurance.duplicated().sum()
# insurance.drop_duplicates()
# insurance.drop_duplicates(inplace=True)
# insurance = insurance.drop_duplicates() # to drop if we have some duplicate values 
# print(x)

# 3.Check outliers
# for i in col:
#     if (insurance[i].dtypes == "int64" or insurance[i].dtypes == "float64"):
#         plt.boxplot(insurance[i])
#         plt.xlabel(i)
        # plt.show()

# treat outlier 
# BMI
# Q1 = insurance.bmi.quantile(0.25)
# Q3 = insurance.bmi.quantile(0.75)
# IQR = Q3-Q1
# lowerFench = Q1-1.5*IQR
# upperFench = Q3+1.5*IQR
# insurance = insurance[(insurance.bmi>=lowerFench)&(insurance.bmi<=upperFench)]

# #Past_consulatations
# Q1 = insurance.past_consultations.quantile(0.25)
# Q3 = insurance.past_consultations.quantile(0.75)
# IQR = Q3-Q1
# lowerFence = Q1-1.5*IQR
# upperFence = Q3+1.5*IQR
# insurance = insurance[(insurance.past_consultations >= lowerFence) & (insurance.past_consultations<=upperFence)]

# #Hospital_expenditure
# Q1 = insurance.Hospital_expenditure.quantile(0.25)
# Q3 = insurance.Hospital_expenditure.quantile(0.75)
# IQR = Q3-Q1
# lowerFence = Q1-1.5*IQR
# upperFence = Q3+1.5*IQR
# insurance = insurance[(insurance.Hospital_expenditure >= lowerFence) & (insurance.Hospital_expenditure<=upperFence)]

# #Anual_Salary
# Q1 = insurance.Anual_Salary.quantile(0.25)
# Q3 = insurance.Anual_Salary.quantile(0.75)
# IQR = Q3-Q1
# lowerFence = Q1-1.5*IQR
# upperFence = Q3+1.5*IQR
# insurance = insurance[(insurance.Anual_Salary >= lowerFence) & (insurance.Anual_Salary<=upperFence)]


#  #NUmber_of_past_hospitalizations
# Q1 = insurance.NUmber_of_past_hospitalizations.quantile(0.25)
# Q3 = insurance.NUmber_of_past_hospitalizations.quantile(0.75)
# IQR = Q3-Q1
# lowerFence = Q1-1.5*IQR
# upperFence = Q3+1.5*IQR
# insurance = insurance[(insurance.NUmber_of_past_hospitalizations >= lowerFence) & (insurance.NUmber_of_past_hospitalizations<=upperFence)]

# this method is time consuming lets automate this with loops
cols=['bmi','past_consultations','Hospital_expenditure','Anual_Salary','NUmber_of_past_hospitalizations']
for i in cols:
    Q1 = insurance[i].quantile(0.25)
    Q3 = insurance[i].quantile(0.75)
    IQR = Q3-Q1
    lowerFence = Q1-1.5*IQR
    upperFence = Q3+1.5*IQR
    insurance = insurance[(insurance[i] >= lowerFence) & (insurance[i]<=upperFence)]

for i in cols:
    if(insurance[i].dtypes=="int64" or insurance[i].dtypes=="float64"):
        plt.boxplot(insurance[i])
        plt.xlabel(i)
        # plt.show()


# here we have do the data cleaning part or EDA part 
# now we are going with ML part 

# First of all we have to import libraries for machine learning 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# print(col)
x = insurance.loc[:,['age',  'bmi', 'children', 'Claim_Amount', 'past_consultations', 'num_of_steps', 'Hospital_expenditure', 'NUmber_of_past_hospitalizations', 'Anual_Salary']] #Independent

# x = pd.get_dummies(x, drop_first=True)
y = insurance.loc[:,'charges'] #Target column
# print(x)
# print(y)


# now divide the data for training and testing 
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.80,random_state=0)  #(independent, dependent or target, train size) random state  = 0 will take the same split all the time 

# Create a linear regression model
my_model = LinearRegression()

# Train the model with the training data
my_model.fit(x_train,y_train)


#Test the model with testing data 
model_answers = my_model.predict(x_test)
# print(model_answers) # predicted answers
# print(y_test) # actual answers 


# compare model_answers with y_test(correct answers)
# compare both values by a table 
result = pd.DataFrame(columns=["Actual Values", "Predicted values"])
result["Actual Values"] = y_test
result["Predicted values"] = model_answers
# print(result)



#   Now i am going to compare these values 
from sklearn.metrics import *
result = r2_score(y_test, model_answers)
# r2_score is a techer which is going to compare correct answer with your model answer
print(result)

sns.regplot(x = model_answers, y = y_test)
plt.xlabel("Predicted charges value")
plt.ylabel("Actual charges value")
plt.title("Regression plot")
# plt.show()


# bcz our charges values are very high thats why this above graph is showing in that why use this instead
plt.figure(figsize=(8,6))
plt.scatter(y_test, model_answers)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')  # 45 degree ideal line

plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.title("Actual vs Predicted")
plt.show()

