import numpy as np

class Linear_Regression:

    def __init__(self, l_rate=0.001, num_iter=500):
        self.l_rate = l_rate
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def fit(self, x,y):
        rows, cols = x.shape
        
        self.weights = np.zeros(cols)
        self.bias = 0

        for i in range(self.num_iter):
            y_pred = np.dot(x,self.weights) + self.bias

            dw = (np.dot(x.T,(y_pred-y)))/rows
            db = (np.sum(y_pred-y))/rows

            self.weights=self.weights-self.l_rate*dw
            self.bias=self.bias-self.l_rate*db
        
    def predict(self,x):
        pred = np.dot(x, self.weights) + self.bias
        return pred