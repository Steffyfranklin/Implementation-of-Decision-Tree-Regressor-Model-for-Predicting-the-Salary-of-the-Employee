# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Steffy Aavlin Raj.F.S
RegisterNumber: 212224040330 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("/content/Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())  

x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("Predicted Salary for [5,6]:", dt.predict([[5, 6]]))

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

![image](https://github.com/user-attachments/assets/3f7ebdba-4441-4652-a529-b7426c40e9c7)

![image](https://github.com/user-attachments/assets/b8186765-7328-4a8d-9fb7-d5b709995c1e)

![image](https://github.com/user-attachments/assets/7975f9d9-a991-4ea0-8574-fd8e57e2c70b)

![image](https://github.com/user-attachments/assets/2038edf2-0e20-453b-ae11-fdd49499a65a)

![image](https://github.com/user-attachments/assets/4e4d03f8-a684-4038-a433-2d5b4d434722)

![image](https://github.com/user-attachments/assets/2a3f1d22-0955-4704-a218-29b3351a9c6a)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
