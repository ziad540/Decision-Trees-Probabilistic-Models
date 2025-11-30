import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature        # Index of the feature to split on
        self.threshold = threshold    # Threshold value for the split
        self.left = left              # Left child node
        self.right = right            # Right child node
        self.value = value            # Class label for leaf nodes
        
        
    def is_leaf(self):
        return self.value is not None    
        





class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.feature_importances_ = None
        
    def fit(self, X, y,depth=0):
        self.n_samples_total = len(y)
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.feature_importances_ = np.zeros(X.shape[1])
        print(f"Fitting tree with X shape: {X.shape}, y shape: {y.shape}, n_features: {self.n_features}")
        self.root = self._grow_tree(X, y)
        print("Tree fitting complete.")
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        ## Check stopping criteria
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            # print(f"Leaf node created at depth {depth}. Value: {leaf_value}, Samples: {n_samples}")
            return Node(value=leaf_value)    
        
        feat_idx = np.random.choice(n_features, self.n_features, replace=False)
        
        ##find best split
        best_threshold, best_feature, best_gain = self.best_split(X, y, feat_idx)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            print(f"No valid split found at depth {depth}. Creating leaf with value: {leaf_value}")
            return Node(value=leaf_value)
            
        # Update feature importance
        self.feature_importances_[best_feature] += (n_samples / self.n_samples_total) * best_gain

        ##create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)        
        
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common    
    
    def best_split(self, X, y, feat_idx):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feature in feat_idx:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold
                    
        return split_threshold, split_idx, best_gain
    
    def _information_gain(self, y, X_column, split_threshold):
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Generate split
        left_idxs, right_idxs = self._split(X_column, split_threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Weighted average child entropy
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        # Information gain is difference in entropy
        ig = parent_entropy - child_entropy
        return ig
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
            
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
