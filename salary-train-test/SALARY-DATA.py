#1 Import Neccessary Libraries

import pandas as pd
import numpy as np
import seaborn as sns

Train = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\blackbox techniqueSVM\Datasets_SVM\SalaryData_Train (1).csv")
Test = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\blackbox techniqueSVM\Datasets_SVM\SalaryData_Test (1).csv")

string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

##Preprocessing the data. As, there are categorical variables
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
for i in string_columns:
        Train[i]= number.fit_transform(Train[i])
        Test[i]=number.fit_transform(Test[i])
        
##Capturing the column names which can help in futher process
colnames = Train.columns
colnames
len(colnames)

x_train = Train[colnames[0:13]]
y_train = Train[colnames[13]]
x_test = Test[colnames[0:13]]
y_test = Test[colnames[13]]

##Normalmization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

x_train = norm_func(x_train)
x_test =  norm_func(x_test)

from sklearn.svm import SVC

model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test) # Accuracy = 80.98%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) # Accuracy = 84.35%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) # Accuracy = 84.32%

#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test) #Accuracy = 84.32%
