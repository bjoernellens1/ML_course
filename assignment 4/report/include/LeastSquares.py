class LinearRegression(Predictor):
    def __init__(self, dataset):
        super().__init__(dataset)
        
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        y = y.values.reshape(-1, 1)
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.coefficients)
    
    def regression_raw(self):
        # copy the values
        X_train = self.X_train_raw
        y_train = self.y_train_raw
        X_test = self.X_test_raw
        y_test = self.y_test_raw

        # implement linear regression

        # Implement the formula for the least-squares regression line
        X_train_T = np.transpose(X_train)
        beta = np.linalg.inv(X_train_T.dot(X_train)).dot(X_train_T).dot(y_train) # these are the weights

        # Train the model on the training set using the least-squares regression line
        y_pred_train = X_train.dot(beta) # the prediction on the train set

        # Evaluate the performance of the model on the testing set using metrics such as mean squared error and R-squared
        y_pred_test = X_test.dot(beta) # the prediction on the test set

        # calculate the mean squared error
        mse = np.mean((y_test - y_pred_test)**2)
        r_squared = 1 - (np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2))

        print('Mean squared error:', mse)
        print('R-squared:', r_squared)

        # that should do

    def regression_pp(self):
        # copy the values
        X_train = self.X_train_pp
        y_train = self.y_train_pp
        X_test = self.X_test_pp
        y_test = self.y_test_pp
        
        # implement linear regression

        # Implement the formula for the least-squares regression line
        X_train_T = np.transpose(X_train)
        beta = np.linalg.inv(X_train_T.dot(X_train)).dot(X_train_T).dot(y_train) # these are the weights

        # Train the model on the training set using the least-squares regression line
        y_pred_train = X_train.dot(beta) # the prediction on the train set

        # Evaluate the performance of the model on the testing set using metrics such as mean squared error and R-squared
        y_pred_test = X_test.dot(beta) # the prediction on the test set

        # calculate the mean squared error
        mse = np.mean((y_test - y_pred_test)**2)
        r_squared = 1 - (np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2))

        print('Mean squared error:', mse)
        print('R-squared:', r_squared)