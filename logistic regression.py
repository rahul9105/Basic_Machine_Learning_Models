import numpy as np

class Logistic_Regression:

    def __init__(self, l_rate=0.001, num_iter=500):
        self.l_rate = l_rate
        self.num_iter = num_iter
        self.weights = None
        self.bias = None
    
    def fit(self, x, y):
        rows, cols = x.shape

        self.weights = np.zeros(cols)
        self.bias = 0

        for i in range(self.num_iter):
            lin_mod = np.dot(x, self.weights) + self.bias
            y_pred = 1/(1 + np.exp(-x))

            dw = (np.dot(x.T,(y_pred-y)))/rows
            db = (np.sum(y_pred-y))/rows

            self.weights = self.weights - self.l_rate*dw
            self.bias = self.bias - self.l_rate*db

    
    def predict(self,x):
        lin_mod = np.dot(x, self.weights)+self.bias
        y_pred = 1/(1 + np.exp(-x))
        y_pred_class = [1 if x<0.5 else 0 for x in y_pred]
        return y_pred_class