import numpy as np
from collections import Counter

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