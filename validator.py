import numpy as np
from classifier import KNN 

class Validator:
    def __init__(self, data, k = 1):
        self.data = data
        self.k = k

    def leave_one_out(self):
        prediction = 0
        total_instances = len(self.data)

        for i in range(total_instances):
            training_data = self.data.drop(i)
            test_instance = self.data.iloc[i]
            classifier = KNN(training_data)
            classifier.train()
            predicted_class = classifier.test(test_instance.drop('class'), self.k)
            actual_class = test_instance['class']

            if predicted_class == actual_class:
                prediction +=1
        
        accuracy = prediction/total_instances
        return accuracy
