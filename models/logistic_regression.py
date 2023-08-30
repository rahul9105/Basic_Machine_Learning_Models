import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

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
            y_pred = 1/(1 + np.exp(-lin_mod))

            dw = (np.dot(x.T,(y_pred-y)))/rows
            db = (np.sum(y_pred-y))/rows

            self.weights = self.weights - self.l_rate*dw
            self.bias = self.bias - self.l_rate*db

    
    def predict(self,x):
        lin_mod = np.dot(x, self.weights)+self.bias
        y_pred = 1/(1 + np.exp(-lin_mod))
        y_pred_class = [1 if x<0.5 else 0 for x in y_pred]
        return y_pred_class

# Driver code
# if __name__ == '__main__':
#     def accuracy(y_true, y_pred):
#         accuracy = np.sum(y_true == y_pred)/len(y_true)
#         return accuracy
    
#     data = datasets.load_breast_cancer()
#     X, y = data.data, data.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1234)

#     clf = Logistic_Regression(l_rate=0.00001, num_iter=50)
#     clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)

#     print("Accuracy:", accuracy(y_test, predictions))