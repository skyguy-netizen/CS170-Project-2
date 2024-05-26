import numpy as np
import sys

class NN:

    # Euclidean distance
    def distance(self, p1, p2):
        distance = 0

        for i in range(0, len(p1)):
            distance = distance + (p1[i]-p2[i+1])**2
        
        distance = distance**0.5
        return distance

    def __init__(self, features, filename):
        file = open(filename, "r")
        datapoints = file.readlines()
        self.data = np.empty((0,len(features)+1))
        features.insert(0, 0)

        # Add certain features of datapoints
        for point in datapoints:
            temp = np.array(point.split()).astype(float)
            temp = temp[features]
            self.data = np.append(self.data, [temp],axis=0)

        file.close()

        # Normalize data
        temp = self.data[:, 1:self.data.shape[1]]
        temp = (temp-np.mean(temp))/np.std(temp)
        self.data[:, 1:self.data.shape[1]] = temp
    
    # Returns class of closest point
    def test(self, point):
        
        # Normalize point
        point = np.array(point)
        point = (point-np.mean(point))/np.std(point)

        min_class = 0
        min_distance = sys.float_info.max
        for data_point in self.data:
            d = self.distance(point, data_point)
            if(d < min_distance):
                min_distance = d
                min_class = data_point[0]
        
        return min_class