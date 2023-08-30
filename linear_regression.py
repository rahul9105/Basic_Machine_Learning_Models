import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

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

# Driver Code
# if __name__ == '__main__':
#     def mse(y_true, y_pred):
#         return np.mean((y_true-y_pred)**2)
    
#     X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1234)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#     fig = plt.figure(figsize=(8,6))
#     plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
#     plt.show()

#     regressor = Linear_Regression(l_rate=0.01, num_iter=1000)
#     regressor.fit(X_train, y_train)
#     predictions = regressor.predict(X_test)

#     print("Accuracy:", mse(y_test, predictions))