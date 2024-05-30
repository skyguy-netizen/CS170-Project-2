import numpy as np
from classifier import NN 

class Validator:
    def __init__(self, data):
        self.data = data

    def leave_one_out(self):
        prediction = 0
        total_instances = len(self.data)

        for i in range(total_instances):
            training_data = self.data.drop(i)
            test_instance = self.data.iloc[i]
            classifier = NN(training_data)
            classifier.train()
            predicted_class = classifier.test(test_instance.drop('class'))
            actual_class = test_instance['class']

            if predicted_class == actual_class:
                prediction +=1
        
        accuracy = prediction/total_instances
        return accuracy
