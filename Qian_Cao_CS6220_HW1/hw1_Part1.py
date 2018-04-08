import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import os
import sys


#load data
def load_csv(file):
    file = os.path.join("HW1_data",file)
    X = np.genfromtxt(file, delimiter=",",skiprows=1)
    return X

#compute weight(theta)
def normal_equation(x,y,lam):
    z = inv(np.dot(x.transpose(), x)+ lam*np.identity(x.shape[1]))
    theta = np.dot(np.dot(z, x.transpose()), y)
    return theta

#compute MSE(cost)
def compute_cost(x,y,theta,lam):
    m = y.shape[0]
    error = np.dot(x,theta) - y
    cost = error.T.dot(error)/float(m)
    cost = cost[0]
    return cost

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    train_matrix = load_csv(file1)
    test_matrix = load_csv(file2)
    Y_training = train_matrix[:,-1][:, np.newaxis]
    X_training = train_matrix[:,:-1]
    #X_training = np.insert(train_matrix[:,:-1], 0, 1, axis=1)
    Y_test = test_matrix[:,-1][:, np.newaxis]
    X_test = test_matrix[:,:-1]
    #X_test = np.insert(test_matrix[:,:-1], 0, 1, axis=1)
    
    #introduce Lambda(a)
    a = np.arange(0,151,1)
    #print(a)
    trainingCostValues = []
    testCostValues = []
    for l in a:
        #cost_training = []
        #cost_test = []
        theta = normal_equation(X_training,Y_training,l)
        #calculate the cost for both training data and test data
        cost1 = compute_cost(X_training,Y_training,theta,l)
        cost2 = compute_cost(X_test,Y_test,theta,l)
        trainingCostValues.append(cost1)
        testCostValues.append(cost2)
        
    #plot the figure
    plt.suptitle("Ridge regression Data plot")
    plt.plot(a,trainingCostValues,'b')
    plt.plot(a,testCostValues,'r')
    plt.ylabel("MSE")
    plt.xlabel("Lambda ")
    fileName = "RidgeRegression1";
    plt.savefig(fileName)
    
    #print Lambda and MSE
    print(a[testCostValues.index(min(testCostValues))])
    print(min(testCostValues))
