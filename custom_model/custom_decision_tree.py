import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# Load and preprocess data
def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Custom Decision Tree implementation
class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Convert y to a numpy array for numpy-style indexing
        y = y.to_numpy() if isinstance(y, pd.Series) else y

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return {'leaf': True, 'value': leaf_value}

        best_feature, best_threshold = self._best_criteria(X, y, n_features)
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return {'leaf': False, 'feature': best_feature, 'threshold': best_threshold, 'left': left, 'right': right}

    def _best_criteria(self, X, y, n_features):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        for feature_idx in range(n_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature_idx], y)))
            thresholds = np.array(thresholds)
            classes = np.array(classes)

            num_left = [0] * len(np.unique(y))
            num_right = [np.sum(classes == c) for c in np.unique(y)]

            for i in range(1, len(thresholds)):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(len(np.unique(y))))
                gini_right = 1.0 - sum((num_right[x] / (len(thresholds) - i)) ** 2 for x in range(len(np.unique(y))))
                gini = (i * gini_left + (len(thresholds) - i) * gini_right) / len(thresholds)
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        return best_feature, best_threshold

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        return self._traverse_tree(x, node['right'])

def main():
    # Load and preprocess the data
    df = load_data('heart.csv')  # Make sure 'heart.csv' is in the same directory as this script
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train, X_test = standardize_data(X_train, X_test)
    
    # Create and train the custom decision tree model
    model = DecisionTree(max_depth=3)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Custom Decision Tree Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
