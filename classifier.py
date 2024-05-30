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
        distances = np.argmin(self.distances(norm_point))
        class_label = self.data.iloc[distances]['class']
        return class_label

    # Euclidean distance
    def distances(self, point):
        return np.sum((self.data.drop(columns = 'class') - point) ** 2, axis = 1)