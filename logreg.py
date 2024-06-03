import pandas as pd
import numpy as np

class LogReg:
    def __init__(self, data):
        self.data = data
        self.data['class'] = self.data['class'].map({2: 1, 1: -1})
        self.X = data.drop(columns = 'class')
        self.Y = data['class']

    def sigmoid(self, f):
        return 1/(1 + np.exp(-f))

    def loss_function(self, w):
        loss = np.sum(-np.log(self.sigmoid(self.Y * (self.X @ w))))
        return loss

    def train(self):   
        X = self.X 
        Y = self.Y   
        m,n = X.shape
        tc = 0.0001
        eta = 1e-5
        w = np.zeros(n)
        prev_loss = float('inf')
        curr_loss = self.loss_function(w)
        losses = []
        while (abs(curr_loss - prev_loss) >= tc):  
            fx = X @ w
            gd = X.T @ ((1 - self.sigmoid(Y * fx)) * Y) 
            w = w + (eta * gd)
            prev_loss = curr_loss
            curr_loss = self.loss_function(w)  
        self.w = w
        return w
        
            
    def test(self, x):
        threshold_p = 0.5
        pred_p = self.sigmoid(x @ self.w)
        pred = np.where(pred_p > threshold_p, 2, 1)
        return pred

    def accuracy(self, X, Y):
        preds_y = self.test(X)
        return (preds_y != Y).mean()