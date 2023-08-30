import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

class K_Nearest_Neighbours:

    def __init__(self, neigh=4):
        self.k = neigh

    def fit(self, x, y):
        self.X = x
        self.Y = y

    def predict(self, X):
        labels = [self.pred_class(x) for x in X]
        return labels

    def pred_class(self,x):
        dist = [np.sqrt(np.sum((x-x1)**2)) for x1 in self.X]
        ind = np.argsort(dist)[:self.k]
        lab = [self.Y[i] for i in ind]
        recur = Counter(lab).most_common(1)
        return recur[0][0]
    
# Driver Code
# if __name__ == '__main__':
#     def accuracy(y_true, y_pred):
#         accuracy = np.sum(y_true == y_pred) / len(y_true)
#         return accuracy
    
#     data = datasets.load_iris()
#     X = data.data
#     y = data.target

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#     plt.figure()
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
#     plt.show()

#     clf = K_Nearest_Neighbours(neigh=3)
#     clf.fit(X_train, y_train)
#     predictions = clf.predict(X_test)
#     print("Accuracy:", accuracy(y_test, predictions))