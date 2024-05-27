import numpy as np
import pandas as pd

class NN:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def train(self):
        (self.data, self.mean, self.std) = self.normalize(self.raw_data.drop(columns = 'class'))
        self.data['class'] = self.raw_data['class']
        
    
    def normalize(self, data):
        data_mean = data.mean(axis = 0)
        data_std = data.std(axis = 0)     

        norm_data = (data - data_mean) / data_std
        return (norm_data, data_mean, data_std)

    # Returns class of closest point
    def test(self, point):
        # Normalize point
        norm_point = (point - self.mean) / self.std
        min_class = None
        min_distance = float('inf')
        for _, row in self.data.iterrows():
            d = self.distance(norm_point, row.drop(columns = 'class'))
            if(d < min_distance):
                min_distance = d
                min_class = row['class']
        return min_class

    # Euclidean distance
    def distance(self, p1, p2):
        distance = 0
        for i in range(len(p1)):
            distance = distance + (p1.iloc[i]-p2.iloc[i])**2
        return distance