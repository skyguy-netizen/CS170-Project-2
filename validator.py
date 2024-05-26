import numpy as np

class validator:
    
    def leave_one_out(data):
        predicted_values = []
        data_classes = [entry[0] for entry in data]
        for index in range(len(data)):
            test_data = data[index]