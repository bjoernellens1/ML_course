import numpy as np

# Implement the rbf_kernel() function
def rbf_kernel(X, centers, gamma):
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    K = np.zeros((n_samples, n_centers))

    for i in range(n_samples):
        for j in range(n_centers):
            diff = X[i] - centers[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))

    return K

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing Prices dataset
data = fetch_california_housing()
X = data['data']
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=None, shuffle=True, stratify=None)

# Standardize the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

# Choose the number of centroids and the RBF kernel width
num_centroids = 100
gamma = 0.1

# Randomly select the centroids from the training set
np.random.seed(42)
idx = np.random.choice(X_train_std.shape[0], num_centroids, replace=False)
centroids = X_train_std[idx]

# Compute the RBF features for the training and testing sets
rbf_train = rbf_kernel(X_train_std, centroids, gamma)
rbf_test = rbf_kernel(X_test_std, centroids, gamma)

# Fit a linear regression model on the original and RBF-transformed data
linreg_orig = LinearRegression().fit(X_train_std, y_train)
linreg_rbf = LinearRegression().fit(rbf_train, y_train)

# Evaluate the models on the testing set
y_pred_orig = linreg_orig.predict(X_test_std)
mse_orig = mean_squared_error(y_test, y_pred_orig)
r2_orig = r2_score(y_test, y_pred_orig)

y_pred_rbf = linreg_rbf.predict(rbf_test)
mse_rbf = mean_squared_error(y_test, y_pred_rbf)
r2_rbf = r2_score(y_test, y_pred_rbf)

# Print the results
print("Linear regression on original data:")
print("MSE:", mse_orig)
print("R^2:", r2_orig)

print("\nLinear regression on RBF-transformed data:")
print("MSE:", mse_rbf)
print("R^2:", r2_rbf)