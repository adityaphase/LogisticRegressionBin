
import pandas as pd
from scipy import optimize as op
import numpy as np
from math import exp

# define sigmoid function
def Sigmoid(z):
    return 1/(1 + np.exp(-z))
#define cost function
def cost(theta,x,y):
    m,n = x.shape #extract the number of rows and columns
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    #doing a dot product of x and theta vectors
    n1 = (np.log(Sigmoid(x.dot(theta)))).reshape((m, 1))
    n2 = (np.log(1-Sigmoid(x.dot(theta)))).reshape((m,1))
    func = y * n1 + (1 - y) * n2
    J = -((np.sum(func))/m)
    return J
#define gradient function
def gradientDescent(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1))
    y = y.reshape((m,1))
    sigmoid_x_theta = Sigmoid(x.dot(theta))
    grad = ((x.T).dot(sigmoid_x_theta-y))/m
    return grad.flatten()
#define a predict function with an accuracy snippet
def predict(X1, y1, modelledTheta):
    c = 0
    for i in range(0, len(X1)):
        z = 1 + modelledTheta[0]*X1[i, 0] + modelledTheta[1]*X1[i, 1]
        y2.append(round(1 / (1 + exp(-z))))
    for i in range(0, len(y1)):
        if y2[i] == y1[i]:
            c += 1
    print("Accuracy: ", (c / len(X1)))
#read and loadout data
dataset = pd.read_csv("/Users/adityaphase/Desktop/ex2data2.csv", usecols=['X1', 'X2', 'Y1'])
x = dataset.iloc[:, [0, 1]]
y = dataset.iloc[:, [2]]
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sx = StandardScaler()
xtrain = sx.fit_transform(xtrain)
xtest = sx.transform(xtest)
X = np.asarray(xtrain)
y = np.asarray(ytrain)
X1 = np.asarray(xtest)
y1 = np.asarray(ytest)
y2 = []
m , n = X.shape
initialTheta = np.zeros(n)
#using newton conjugate gradient to find minima
Result = op.minimize(fun = cost, x0 = initialTheta, args = (X, y), method = 'TNC', jac = gradientDescent)
modelledTheta = Result.x
print(modelledTheta)
predict(X1, y1, modelledTheta)
