import numpy as np
import matplotlib.pyplot as plt


def handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def nordata(data):
    mean = data.mean(axis=0)
    var = data.std(axis=0)
    var = handle_zeros_in_scale(var)
    nordata = np.divide(data - mean, var)
    return nordata


def loadDataSet():
    arffFile = open('segment.arff', 'r')
    data = []
    for line in arffFile.readlines():
        if not (line.startswith('@')):
            if not (line.startswith('%')):
                if line != '\n':
                    L = line.strip('\n')
                    k = L.split(',')
                    data.append(k)
    data = np.array(data)
    data = data[:, 0:-1].astype(np.float)
    nor_data = nordata(data)
    return nor_data


# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))


# init centroids with 300 indices
def initCentroids(dataSet, k, t):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    L = [773, 1010, 240, 126, 319, 1666, 1215, 551, 668, 528, 1060, 168, 402, 80, 115, 221, 242, 1951, 1725, 754, 1469,
         135, 877, 1287, 645, 272, 1203, 1258, 1716, 1158, 586, 1112, 1214, 153, 23, 510, 05, 1254, 156, 936, 1184,
         1656, 244, 811, 1937, 1318, 27, 185, 1424, 190, 663, 1208, 170, 1507, 1912, 1176, 1616, 109, 274, 1, 1371, 258,
         1332, 541, 662, 1483, 66, 12, 410, 1179, 1281, 145, 1410, 664, 155, 166, 1900, 1134, 1462, 954, 1818, 1679,
         832, 1627, 1760, 1330, 913, 234, 1635, 1078, 640, 833, 392, 1425, 610, 1353, 1772, 908, 1964, 1260, 784, 520,
         1363, 544, 426, 1146, 987, 612, 1685, 1121, 1740, 287, 1383, 1923, 1665, 19, 1239, 251, 309, 245, 384, 1306,
         786, 1814, 7, 1203, 1068, 1493, 859, 233, 1846, 1119, 469, 1869, 609, 385, 1182, 1949, 1622, 719, 643, 1692,
         1389, 120, 1034, 805, 266, 339, 826, 530, 1173, 802, 1495, 504, 1241, 427, 1555, 1597, 692, 178, 774, 1623,
         1641, 661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356, 541, 1455, 741, 583, 1464, 209, 1615, 475, 1903,
         555, 1046, 379, 1938, 417, 1747, 342, 1148, 1697, 1785, 298, 185, 1145, 197, 1207, 1857, 158, 130, 1721, 1587,
         1455, 190, 177, 1345, 166, 1377, 1958, 1727, 1134, 1953, 1602, 114, 37, 164, 1548, 199, 1112, 128, 167, 102,
         87, 25, 249, 1240, 1524, 198, 111, 1337, 1220, 1513, 1727, 159, 121, 1130, 1954, 1561, 1260, 150, 1613, 1152,
         140, 1473, 1734, 137, 1156, 108, 110, 1829, 1491, 1799, 174, 847, 177, 1468, 97, 1611, 1706, 1123, 79, 171,
         130, 100, 143, 1641, 181, 135, 1280, 1442, 1188, 133, 99, 186, 1854, 27, 160, 130, 1495, 101, 1411, 814, 109,
         95, 111, 1582, 1816, 170, 1663, 1737, 1710, 543, 1143, 1844, 159, 48, 375, 1315, 1311, 1422]

    for i in range(k):
        index = L[i + k * t] - 1
        centroids[i, :] = dataSet[index, :]
    return centroids


# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    SSE_list = []
    for t in range(25):
        ## step 1: init centroids
        centroids = initCentroids(dataSet, k, t)
        clusterAssment = np.mat(np.zeros((numSamples, 2)))
        Clusterdistance = np.zeros((numSamples,))
        clusterChanged = True
        count = 0
        while clusterChanged:

            clusterChanged = False
            ## for each sample
            for i in range(numSamples):
                minDist = 100000.0
                minIndex = 0
                ## for each centroid
                ## step 2: find the closest centroid
                for j in range(k):
                    distance = euclDistance(centroids[j, :], dataSet[i, :])
                    if distance < minDist:
                        minDist = distance
                        minIndex = j

                ## step 3: update its cluster
                if k == 1:
                    clusterAssment[i, :] = minIndex, minDist ** 2
                    clusterChanged = False

                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                    clusterAssment[i, :] = minIndex, minDist ** 2

            ## step 4: update centroids
            for j in range(k):
                pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
                centroids[j, :] = np.mean(pointsInCluster, axis=0)

            # step 5: update distance to centroids
            for i in range(numSamples):
                for j in range(k):
                    if clusterAssment[i, 0] == j:
                        Clusterdistance[i] = euclDistance(centroids[j, :], dataSet[i, :])

            if count == 50:
                clusterChanged = False

            count = count + 1

        SSE = np.sum(Clusterdistance**2)
        SSE_list.append(SSE)
    # print 'k=' + str(k) + ':', SSE_list
    return np.array(SSE_list)


if __name__ == "__main__":
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    data = loadDataSet()
    mean_SSE_list = []
    var_SSE_list = []
    for i in range(max(k)):
        SSE = kmeans(data, k[i])
        mean_SSE = SSE.mean()
        var_SSE = SSE.std()
        mean_SSE_list.append(mean_SSE)
        var_SSE_list.append(var_SSE)
        print 'k=', k[i], 'uk=', mean_SSE, 'uk-2std=', mean_SSE - 2 * var_SSE, 'uk+2td=', mean_SSE + 2 * var_SSE

    plt.errorbar(k, mean_SSE_list, yerr=[x * 2 for x in var_SSE_list])
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.xlim(0, max(k) + 1)
    fileName = "costFunction"
    plt.savefig(fileName)
