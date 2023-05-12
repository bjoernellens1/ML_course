class RidgeRegression(Predictor):
    def __init__(self, dataset, alpha=1):
        super().__init__(dataset) # this command initializes the parent class (Predictor) and passes the dataset.
        self.alpha = alpha
    
    def regression_raw(self):
        """
        Fits a ridge regression model on the training data using the specified regularization parameter alpha.
        Using raw dataset
        """
        X_train = self.X_train_raw
        X_test = self.X_test_raw
        y_train = self.y_train_raw
        y_test = self.y_test_raw
        alpha = self.alpha
        
        # Add a cloumn of 1s to the training data to have the correct dimension.
        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        n_features = X_train.shape[1]
        I = np.eye(n_features)
        w = np.linalg.inv(X_train.T.dot(X_train) + alpha * I).dot(X_train.T).dot(y_train)

        #X_test = (X_test - X.mean()) / X.std() #this should not be needed, since the normalization is happening in the constructor
        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        y_pred = X_test.dot(w)

        # calculate the mean squared error
        mse = np.mean((y_test - y_pred)**2)
        r_squared = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))

        # calculate root mean squared error (RMSE)
        rmse = np.sqrt(mse)

        print("Predicted target value:", y_pred[0])
        print("Mean squared error (MSE):", mse)
        print("R-squared:", r_squared)
        print("Root mean squared error (RMSE):", rmse)

    def regression_pp(self):
        """
        Fits a ridge regression model on the training data using the specified regularization parameter alpha.
        Using processed dataset.
        """
        X_train = self.X_train_pp
        X_test = self.X_test_pp
        y_train = self.y_train_pp
        y_test = self.y_test_pp
        alpha = self.alpha
        
        # Add a cloumn of 1s to the training data to have the correct dimension.
        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        n_features = X_train.shape[1]
        I = np.eye(n_features)
        w = np.linalg.inv(X_train.T.dot(X_train) + alpha * I).dot(X_train.T).dot(y_train)

        X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        y_pred = X_test.dot(w)

        # calculate the mean squared error
        mse = np.mean((y_test - y_pred)**2)
        r_squared = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))

        # calculate root mean squared error (RMSE)
        rmse = np.sqrt(mse)

        print("Predicted target value:", y_pred[0])
        print("Mean squared error (MSE):", mse)
        print("R-squared:", r_squared)
        print("Root mean squared error (RMSE):", rmse)