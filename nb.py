import pandas as pd

class NB:
    def __init__(self, data):
        self.data = data
    
    def distribution(self, counts):
        total = sum(counts)
        percentages = [count / total for count in counts]
        return percentages

    def train(self):
        X = self.data.drop(columns = 'class')
        Y = self.data[['class']]
        maxfeatval = X.max().max()
        priorp = np.array([0.0, 0.0])
        priorp[1] = np.count_nonzero(Y == 2) / len(Y) 
        priorp[0] = 1 - priorp[1]
        
        n = X.shape[1]
        condp = np.zeros((n, int(maxfeatval + 1), 2) )

        for i in range(n):
            c1 = X[Y == 1, i] #extract all data where Y == 0
            c2 = X[Y == 2, i] #extract all data where Y == 1

            class0_probs = distribution(np.array([np.count_nonzero(c1 == x) for x in range(int(maxfeatval + 1))]) + 1) #calculate the probability for each feature for class 0...added smoothing
            class1_probs = distribution(np.array([np.count_nonzero(c2 == x) for x in range(int(maxfeatval + 1))]) + 1) #calculate the probability for each feature for class 1...added smoothing

            
            condp[i, :, 0] = class0_probs #assign the probs to respective slot in condp
            condp[i, :, 1] = class1_probs #same as above

        #Just to check if correct shape: print(condp.shape)
        return (priorp, condp)  ## or whatever they are named in your code