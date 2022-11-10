#1 Import Neccessary Libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Import Data

forestdata = pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\blackbox techniqueSVM\Datasets_SVM\forestfires.csv")
forestdata

forestdata.columns

#initial Analysis
forestdata.dtypes

forestdata.isna().sum()

sns.countplot(data=forestdata, y=forestdata['month'])
plt.show()
#### from the above plot we can see that the max fire comes in the forest in the month of August and September, March 
#### As compare to the August , September & March the August, september are contributing more majority of fire into the forest 
#### Main reason behind it is the deforestation and cutting of the trees . Due to cutting the trees it brings the change in the climate and rise in the humidity 

sns.countplot(data=forestdata, y=forestdata['day'])
plt.show()
####### From the above plot we can see that the The Majority of the fire happens in the month of Aug then Sep and during sundays the count is high followed by saturday and friday

forestdata.shape

data = forestdata.describe()

##Dropping the month and day columns
forestdata.drop(["month","day"],axis=1,inplace =True)

##Normalising the data as there is scale difference
predictors = forestdata.iloc[:,0:28]
target = forestdata.iloc[:,28]

def norm_func(i):
    x= (i-i.min())/(i.max()-i.min())
    return (x)

fires = norm_func(predictors)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25, stratify = target)

model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test) # Accuracy = 94.61%

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) #Accuacy = 77.69%

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) #Accuracy = 74.61%

#'sigmoid'
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(x_train,y_train)
pred_test_sig = model_rbf.predict(x_test)

np.mean(pred_test_sig==y_test) #Accuracy = 74.61%
