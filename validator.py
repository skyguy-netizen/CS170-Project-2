import numpy as np
from classifier import classifier
class validator:
    
    def __init__(self,classifier,dataset):
        self.classifier = classifier
        self.dataset = dataset

    def leave_one_out(self,feature_subset):
        prediction = 0
        total_instances = len(self.data)

        for i in range(total_instances):
            training_data = np.delete(self.data,i,axis=0)
            test_instance = self.data[i]
            train_classifier = NN(features=feature_subset,filename=None)
            train_classifier.data = training_data

            predicted_class = train_classifier.test(test_instance[1:])
            actual_class = test_instance[0]

            if predicted_class == actual_class:
                prediction +=1
        
        accuracy = prediction/total_instances
        return accuracy
