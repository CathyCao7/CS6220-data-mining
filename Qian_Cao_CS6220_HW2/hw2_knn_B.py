import numpy as np

def centered_train(matrix):
    a = matrix
    n = a.shape[0]
    mean_list = []
    res = np.empty(shape=[a.shape[0],0])
    for i in range(a.shape[1]):
        mean_list.append(np.mean(a[:,i]))
    diff = np.tile(mean_list,(n,1)) - a
    squareddiff = diff ** 2
    squareddiv = np.sum(squareddiff, axis = 0)/n
    divide = squareddiv ** 0.5
    for i in range(a.shape[1]):
        col = matrix[:, i]
        mean_val = mean_list[i]
        div_val = divide[i]
        b = np.apply_along_axis(lambda x: ((x-mean_val)/div_val),0,col)
        res = np.insert(res,res.shape[1],values=b.flatten(),axis=1)
    return res,mean_list,divide

def centered_test(matrix,mean_list,divide):
    a = matrix
    res = np.empty(shape=[a.shape[0],0])
    for i in range(a.shape[1]):
        col = matrix[:, i]
        mean_val = mean_list[i]
        div_val = divide[i]
        b = np.apply_along_axis(lambda x: ((x-mean_val)/div_val),0,col)
        res = np.insert(res,res.shape[1],values=b.flatten(),axis=1)
    return res

def loadDataSet():
    train_data = np.genfromtxt("spam_train.csv",delimiter=",",skip_header=1)
    test_data = np.genfromtxt("spam_test.csv",delimiter=",",skip_header=1)
    train_label = train_data[:,-1]
    train_data = train_data[:,1:-1]
    test_label = test_data[:,-1]
    test_data = test_data[:,1:-1]
    #if run (b) with z-score normalization
    train_data,me,div = centered_train(train_data)
    test_data = centered_test(test_data,me,div)
    return train_data,train_label,test_data,test_label

def kNNClassify(newInput, dataSet, label, k):  
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row
    diff = np.tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise  
    squaredDiff = diff ** 2 # squared for the subtract  
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row  
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)  
    classCount = {} # define a dictionary
    for i in range(k):
        voteLabel = label[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)+1
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

def testAccuracy():
    k = [1, 5, 11, 21, 41, 61, 81, 101, 201, 401]
    train_x, train_y, test_x, test_y = loadDataSet()  
    numTestSamples = test_x.shape[0]
    for n in range(10):
        matchCount = 0
        for i in range(numTestSamples):
            predict = kNNClassify(test_x[i], train_x, train_y, k[n])
            if predict == test_y[i]:
                matchCount += 1
        accuracy = float(matchCount) / numTestSamples
        print(accuracy)

if __name__ == '__main__':
    testAccuracy()
