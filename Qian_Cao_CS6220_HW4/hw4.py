import numpy as np
import copy

def nordata(data):
    mean = data.mean(axis=0)
    var = data.std(axis=0)
    #var = handle_zeros_in_scale(var)
    nordata = np.divide(data - mean, var)
    return nordata

def numlabel(label):
    num_label=[]
    for i in range(len(label)):
        if label[i] == 'car':
            num_l = 1
        else:
            num_l = 0
        num_label.append(num_l)
    return num_label
    
def loadDataSet():
    arffFile = open('veh-prime.arff', 'r')
    data = []
    for line in arffFile.readlines():
        if not (line.startswith('@')):
            if not (line.startswith('%')):
                if line != '\n':
                    L = line.strip('\n')
                    k = L.split(',')
                    data.append(k)
    data = np.array(data)
    label = data[:,-1]
    numeric_label = numlabel(label)
    data = data[:, 0:-1].astype(np.float)
    nor_data = nordata(data)
    return nor_data,numeric_label

def correlation(training_data,training_label,feature):
    length = float(len(training_data))
    sum_sq_x = 0.0
    sum_sq_y = 0.0
    sum_coproduct = 0.0
    mean_x = 0.0
    mean_y = 0.0
    for i in range(len(training_data)):
        sum_sq_x += training_data[i][feature]**2
        sum_sq_y += float(training_label[i])**2
        sum_coproduct += training_data[i][feature] * float(training_label[i])
        mean_x += training_data[i][feature]
        mean_y += float((training_label[i]))
    mean_x = mean_x / length
    mean_y = mean_y / length
    pop_sd_x = ((sum_sq_x/length) - (mean_x **2))**0.5
    pop_sd_y = ((sum_sq_y/length) - (mean_y **2))**0.5
    cov_x_y = (sum_coproduct/length)- (mean_x * mean_y)
    correlation = cov_x_y / (pop_sd_x * pop_sd_y)
    return abs(correlation)

def kNNClassify(newInput, dataSet, label, k):  
    numSamples = len(dataSet) # shape[0] stands for the num of row
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
    
def filter(training_data,training_label,output_filter):
    correlations = []
    order = []
    for j in range(len(training_data[0])):
        correlations.append(correlation(training_data,training_label,j))
    find_order = list(correlations)
    for i in range(len(correlations)):
        maximum = max(find_order)
        order.append(find_order.index(maximum))
        find_order[order[i]] = -1

    text = "Filter Method\n"
    text += "Part A: Features listed in descending order according to the |r| value\n"
    output_filter.write(text)
    for i in range(len(correlations)):
        text = "Feature "
        text += str(order[i]+1)
        text += " has an |r| of "
        text += str(correlations[order[i]])
        text += str('\n')
        output_filter.write(text)
    text = "\nPart B: Values of m and Avg LOOCV accuracy\n"
    output_filter.write(text)
    for columnnums in range(len(order)):
        columns = []
        total = 0
        sum = 0
        for y in range(columnnums+1):
            columns.append(order[y])
        for i in range(len(training_data)):
            new_training_data = []
            new_training_label = []
            #Leave out this data point(item)
            for x in range(len(training_data)):
                store_row = []
                for columnrange in range(len(columns)):
                    store_row.append(training_data[x][order[columnrange]])
                new_training_data.append(store_row)
                new_training_label.append(training_label[x])
            x =new_training_data.pop(i)
            y =new_training_label.pop(i)
            predict = kNNClassify(x,new_training_data,new_training_label,7)
            total = total+1
            if int(predict) == int(y):
                sum = sum + 1
        text = "M: "
        text += str(columnnums+1)
        text += ", LOOCV Accuracy: "
        text += str(sum)
        text += "/"
        text += str(total)
        text += ", "
        text += str(round(float(sum)*100/float(total),1))
        text += "% Correctly Classified\n"
        output_filter.write(text)

def wrapper(training_data,training_label, output_wrapper, remaining_features):
    text = "Wrapper method\n"
    text += "Iteration 0, Selected Features : {} LOOCV Accuracy: 0/"
    text += str(len(training_data))
    text += "\n"
    output_wrapper.write(text)

    added_features = []
    incorrect = len(training_data)
    iteration = 1
    while incorrect >= 0:
        incorrect = wrapper_helper(training_data,training_label,output_wrapper,incorrect, added_features,remaining_features,iteration)
        iteration = iteration +1 


def wrapper_helper(training_data,training_label, output_wrapper, incorrect, added_features, remaining_features,iteration):
    sums = []
    for columnnums in range(len(remaining_features)):
        sum = 0
        total = 0
        columns = []
        columns = copy.deepcopy(added_features)
        columns.append(remaining_features[columnnums])

        for i in range(len(training_data)):
            new_training_data = []
            new_training_label = []
            #Leave out this data point(item)
            for x in range(len(training_data)):
                store_row = []
                for columnrange in range(len(columns)):
                    store_row.append(training_data[x][columns[columnrange]])
                new_training_data.append(store_row)
                new_training_label.append(training_label[x])
            x =new_training_data.pop(i)
            y =new_training_label.pop(i)
            predict = kNNClassify(x,new_training_data,new_training_label,7)
            total = total+1
            if int(predict) == int(y):
                sum = sum + 1
        sums.append(sum)
    if len(sums) == 0:
        return -1
    maximum = max(sums)

    if incorrect > (total - maximum):
        added_features.append(remaining_features[sums.index(max(sums))])
        remaining_features.pop(sums.index(max(sums)))
        text = "Iteration "
        text += str(iteration)
        text += " Selected Features : { "
        for value in range(len(added_features)):
            num = int(added_features[value])
            num = num + 1
            text += str(num)
            text += " "
        text += "} LOOCV Accuracy: "
        text += str(max(sums))
        text += "/"
        text += str(total)
        text += "\n"
        output_wrapper.write(text)
        return total-maximum
    elif (total-maximum) >= incorrect:
        return -1

def main():
    training_data,training_label = loadDataSet()
    output_filter = open('output_filter.txt', 'w')
    output_wrapper = open('output_wrapper.txt', 'w')
    
    filter(training_data,training_label, output_filter)
    
    remaining_features = list(range(len(training_data[0])))
    wrapper(training_data,training_label, output_wrapper,remaining_features)

if __name__ == "__main__":
    main()
