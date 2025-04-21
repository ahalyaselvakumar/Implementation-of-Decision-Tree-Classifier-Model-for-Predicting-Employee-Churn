# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Dataset
2.Data Preprocessing 
3.Feature and Target Selection 
4.Split the Data into Training and Testing Sets 
5.Build and Train the Decision Tree Model
6.Make Predictions
7.Evaluate the Model

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: AHALYA S
RegisterNumber:  212223230006
*/
```

```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head() #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```


## Output:
![decision tree classifier model](sam.png)

![image](https://github.com/user-attachments/assets/677ad70a-ac17-4c48-8b94-e01f4eaab9e6)


![image](https://github.com/user-attachments/assets/dbc0e4d5-c9bf-484a-810f-d48ad743d126)


![image](https://github.com/user-attachments/assets/4172ad4d-e199-4ce5-bdd3-32bf19a92689)


![image](https://github.com/user-attachments/assets/43129d7e-d3ad-49e3-99ca-64430ab3890b)


![image](https://github.com/user-attachments/assets/072c5220-ea3e-4f91-be95-af833e73fae7)


![image](https://github.com/user-attachments/assets/edc15883-f09a-45cd-96f0-2a5c7c49a92c)


![image](https://github.com/user-attachments/assets/3214d931-fbcf-4e74-a173-a501b0af8978)


![image](https://github.com/user-attachments/assets/ee5e330a-3dd6-4005-8267-7c3ec828dd3e)


![image](https://github.com/user-attachments/assets/0bbbdb29-725b-411c-bffd-87b4bebb2320)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
