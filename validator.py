import numpy as np
from classifier import KNN 
# from logreg import LogReg

class Validator:
    def __init__(self, data, k = 1):
        self.data = data
        self.k = k

    def leave_one_out(self):
        prediction = 0
        total_instances = len(self.data)

        if self.data.shape[1] == 1 and 'class' in self.data.columns:
            classes, counts = np.unique(self.data['class'], return_counts=True)
            def_rate = np.max(counts)/total_instances
            return def_rate

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

    # def test_logreg(self):
    #     prediction = 0
    #     total_instances = len(self.data)
    #     split = int(0.7 * total_instances)
    #     train_data = self.data[:split]
    #     test_data = self.data[split:]
    #     model = LogReg(train_data)
    #     model.train()
    #     xtest = test_data.drop(columns = 'class')
    #     ytest = test_data['class']
    #     accuracy = model.accuracy(xtest, ytest)
    #     return accuracy
