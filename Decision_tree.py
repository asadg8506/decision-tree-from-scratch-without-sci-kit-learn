here is the script download the dataset from readme.md and add your own file path


import numpy as np
import pandas as pd

file_path = r"C:\Users\Gujjar\Desktop\sales_data_sample.csv"
df = pd.read_csv(file_path)

features = ['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'YEAR_ID']
target = 'SALES'

df[features] = df[features].fillna(df[features].mean())

X = df[features].to_numpy()
y = df[target].to_numpy()

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Decision Tree implementation
class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)
    
    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(set(y)) == 1:
            return np.mean(y)
        
        feature_idx = np.random.randint(0, X.shape[1])
        threshold = np.median(X[:, feature_idx])
        
        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold
        
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return np.mean(y)
        
        return {
            'feature': feature_idx,
            'threshold': threshold,
            'left': self._grow_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        }
    
    def predict_sample(self, x, node):
        if isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                return self.predict_sample(x, node['left'])
            else:
                return self.predict_sample(x, node['right'])
        else:
            return node
    
    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])

# Train Single Decision Tree
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

# Predict using Decision Tree
predictions = tree.predict(X_test)


mse = np.mean((y_test - predictions) ** 2)

ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - predictions) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
