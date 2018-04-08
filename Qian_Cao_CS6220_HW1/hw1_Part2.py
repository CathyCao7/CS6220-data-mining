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

#generate random subsets
def generate_subset(X,n):
    training_x_list = []
    training_y_list = []
    for i in range(10):
        a = np.random.random_integers(X.shape[0]-n)
        X_training = X[a:a+n,:]
        y_training = X_training[:, -1][:, np.newaxis]
        X_training = X_training[:,:-1]
        #X_training = np.insert(X_training[:,:-1],0,1,axis=1)
        training_x_list.append(X_training)
        training_y_list.append(y_training)
    return training_x_list,training_y_list

#compute weight(theta)
def normal_equation(x,y,lam):
    z = inv(np.dot(x.transpose(), x)+ lam*np.identity(x.shape[1]))
    theta = np.dot(np.dot(z, x.transpose()), y)
    return theta

#compute MSE(cost)
def compute_cost(x,y,theta):
    m = y.shape[0]
    error = np.dot(x,theta) - y
    cost = error.T.dot(error)/float(m)
    cost = cost[0]
    return cost


if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    a = int(sys.argv[3])
    train_matrix = load_csv(file1)
    test_matrix = load_csv(file2)
    Y_test = test_matrix[:,-1][:, np.newaxis]
    X_test = test_matrix[:,:-1]
    #X_test = np.insert(test_matrix[:,:-1], 0, 1, axis=1)
    
    #change value of Lambda(a)
    #a = 1
    #introduce size of training data(N)
    N = np.arange(1,train_matrix.shape[0],1)
    trainingCostValues = []
    testCostValues = []
    for l in N:
        X_training,Y_training = generate_subset(train_matrix,l)
        cost_training = []
        cost_test = []
        for i in range(10):
            theta = normal_equation(X_training[i],Y_training[i],a)
            #calculate the cost for both training data and test data
            cost1 = compute_cost(X_training[i],Y_training[i],theta)
            cost2 = compute_cost(X_test,Y_test,theta)
            cost_training.append(cost1)
            cost_test.append(cost2)
        meanTrainingCost = sum(cost_training)/float(10)
        trainingCostValues.append(meanTrainingCost)
        meanTestingCost = sum(cost_test)/float(10)
        testCostValues.append(meanTestingCost)
        
    #plot the figure
    plt.suptitle("Learning Curve plot")
    plt.plot(N,trainingCostValues,'b')
    plt.plot(N,testCostValues,'r')
    plt.ylabel("MSE")
    plt.xlabel("N")
    fileName = "LearningCurve";
    plt.savefig(fileName)

    print(min(testCostValues))
