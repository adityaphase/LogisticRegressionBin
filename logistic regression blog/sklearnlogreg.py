import numpy as np
import pandas as pd
dataset = pd.read_csv("/Users/adityaphase/Desktop/ex2data1.csv")
x = dataset.iloc[:, [0, 1]].values
y = dataset.iloc[:, [2]].values
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split( x, y, test_size = 0.25, random_state = 0 )
from sklearn.preprocessing import StandardScaler
sx = StandardScaler()
xtrain = sx.fit_transform(xtrain)
xtest = sx.transform(xtest)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', random_state = 0)
lr.fit( xtrain, ytrain.ravel())
predict = lr.predict(xtest)
from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(ytest, predict))
