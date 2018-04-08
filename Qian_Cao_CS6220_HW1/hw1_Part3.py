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

#CV
#divide the dataset into 90% training and 10% test
def generate_set(X):
    num_test = round(0.1*(X.shape[0]))
    start = 0
    end = num_test
    test_x_list =[]
    test_y_list =[]
    training_x_list = []
    training_y_list = []
    for i in range(10):
        X_test = X[start:end , :]
        tmp1 = X[:start, :]
        tmp2 = X[end:, :]
        X_training = np.concatenate((tmp1, tmp2), axis=0)
        y_training = X_training[:, -1][:, np.newaxis]
        y_test = X_test[:, -1][:, np.newaxis]
        X_training = X_training[:,:-1]
        X_test = X_test[:,:-1]
        #X_training = np.insert(X_training[:,:-1],0,1,axis=1)
        #X_test = np.insert(X_test[:,:-1],0,1,axis=1)
        test_x_list.append(X_test)
        test_y_list.append(y_test)
        training_x_list.append(X_training)
        training_y_list.append(y_training)
        start = end
        end = end+num_test
    return test_x_list,test_y_list,training_x_list,training_y_list


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
    file = sys.argv[1]
    train_matrix = load_csv(file)
    X_test,Y_test,X_training,Y_training = generate_set(train_matrix)
    #introduce Lambda(a)
    a = np.arange(0,151,1)
    #print(a)
    trainingCostValues = []
    testCostValues = []
    for l in a:
        cost_training = []
        cost_test = []
        for i in range(10):
            theta = normal_equation(X_training[i],Y_training[i],l)
            #calculate the cost for both training data and test data
            cost1 = compute_cost(X_training[i],Y_training[i],theta,l)
            cost2 = compute_cost(X_test[i],Y_test[i],theta,l)
            cost_training.append(cost1)
            cost_test.append(cost2)
        meanTrainingCost = sum(cost_training)/float(10)
        trainingCostValues.append(meanTrainingCost)
        meanTestingCost = sum(cost_test)/float(10)
        testCostValues.append(meanTestingCost)

        
    #plot the figure
    plt.suptitle("Ridge regression Data plot")
    plt.plot(a,trainingCostValues,'b')
    plt.plot(a,testCostValues,'r')
    plt.ylabel("MSE")
    plt.xlabel("Lambda ")
    fileName = "RidgeRegression";
    plt.savefig(fileName)
    
    #print Lambda and MSE
    print(a[testCostValues.index(min(testCostValues))])
    print(min(testCostValues))
