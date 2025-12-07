from sklearn.utils import resample
from decisionTree import DecisionTree
import numpy as np
from collections import Counter


class RandomForest:
    def __init__(
        self, n_trees=100, max_depth=None,  min_samples_split=2, n_features=None
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):

        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree_seed = np.random.randint(0, 10000)
            tree = DecisionTree(
                max_depth=self.max_depth,
                n_features=self.n_features,
                min_samples_split=self.min_samples_split,
                random_state=tree_seed,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
       
        X = np.array(X)
        preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(preds, 0, 1)
        y_pred = np.array(
            [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        )
        return y_pred

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
