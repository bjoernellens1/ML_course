import numpy as np
from itertools import combinations_with_replacement

# Implement the polynomial_features() function
def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    polynomial_X = np.ones((n_samples, 1))

    for d in range(1, degree + 1):
        combinations = combinations_with_replacement(range(n_features), d)
        for comb in combinations:
            poly_features = np.prod(X[:, comb], axis=1, keepdims=True)
            polynomial_X = np.hstack((polynomial_X, poly_features))

    return polynomial_X

# Non-linear feature transformation
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load the concrete compressive strength dataset
df = pd.read_excel('Concrete_Data.xls')

# Selecting the Concrete compressive strength as targets, rest is features
X = df.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)
y = df['Concrete compressive strength(MPa, megapascals) ']


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=None, shuffle=True, stratify=None)

# transform the features into second degree polynomial features
X_train_poly_custom = polynomial_features(X_train.values, degree=2)
X_test_poly_custom = polynomial_features(X_test.values, degree=2)

# Train the model
lr_poly_custom = LinearRegression()
lr = LinearRegression()
# fit the model
lr_poly_custom.fit(X_train_poly_custom, y_train)
lr.fit(X_train, y_train)
# predict values from the polynomial transformed features
predictions_poly_custom_train = lr_poly_custom.predict(X_train_poly_custom)
predictions_poly_custom = lr_poly_custom.predict(X_test_poly_custom)
# predict values from the original features
predictions_train = lr.predict(X_train)
predictions = lr.predict(X_test)

# mean squared error
print("Mean squared error (train poly custom): {:.2f}".format(mean_squared_error(y_train, predictions_poly_custom_train)))
print("Mean squared error (test poly custom): {:.2f}".format(mean_squared_error(y_test, predictions_poly_custom)))
print("Mean squared error (train): {:.2f}".format(mean_squared_error(y_train, predictions_train)))
print("Mean squared error (test): {:.2f}".format(mean_squared_error(y_test, predictions)))

# coefficient of determination (R^2)
print("R^2 (train poly custom): {:.2f}".format(r2_score(y_train, predictions_poly_custom_train)))
print("R^2 (test poly custom): {:.2f}".format(r2_score(y_test, predictions_poly_custom)))
print("R^2 (train): {:.2f}".format(r2_score(y_train, predictions_train)))
print("R^2 (test): {:.2f}".format(r2_score(y_test, predictions)))

