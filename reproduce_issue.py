import numpy as np
from decisionTree import DecisionTree
from sklearn.model_selection import train_test_split
import sklearn.datasets

def accuracy(y_true, y_pred):
    if y_pred is None:
        return 0.0
    return np.sum(y_true == y_pred) / len(y_true)

def test():
    print("Loading data...")
    cancer_data = sklearn.datasets.load_breast_cancer()
    X, Y = cancer_data.data, cancer_data.target

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, shuffle=True)

    print("Training Decision Tree...")
    clf = DecisionTree(max_depth=2, min_samples_split=2)
    clf.fit(X_train, Y_train)

    print("Predicting...")
    y_pred_val = clf.predict(X_val)
    print(f"Predictions: {y_pred_val}")

    acc = accuracy(Y_val, y_pred_val)
    print(f"Validation Accuracy: {acc}")

if __name__ == "__main__":
    test()
