class LassoRegression(Predictor):
    def __init__(self, dataset, alpha=1, max_iter=1000, tol=0.0001):
        super().__init__(dataset)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def regression_raw(self):
        """
        Fits a lasso regression model on the training data using the specified regularization parameter alpha, iterations, and tolerance.
        Using raw dataset.
        """
        # Define hyperparameters
        alpha = self.alpha  # regularization strength
        max_iterations = self.max_iter  # number of gradient descent iterations
        tolerance = self.tol

        # Load the data
        diabetes = pd.read_csv("diabetes.csv")

        diabetes.insert(0, "Intercept", 1)

        train_size = int(0.8 * len(diabetes))
        
        X_train = diabetes.iloc[:train_size, :-1].values
        y_train = diabetes.iloc[:train_size, -1].values
        X_test = diabetes.iloc[train_size:, :-1].values
        y_test = diabetes.iloc[train_size:, -1].values

        theta_lasso = np.zeros(X_train.shape[1])
        for i in range(max_iterations):
            theta_prev = theta_lasso.copy()
            for j in range(X_train.shape[1]):
                if j == 0:
                    theta_lasso[j] = np.mean(y_train)
                else:
                    xj = X_train[:, j]
                    rj = y_train - X_train @ theta_lasso + xj * theta_lasso[j]
                    zj = xj @ xj
                    if zj == 0:
                        theta_lasso[j] = 0
                    else:
                        if np.sum(xj * rj) > alpha / 2:
                            theta_lasso[j] = (np.sum(xj * rj) - alpha / 2) / zj
                        elif np.sum(xj * rj) < - alpha / 2:
                            theta_lasso[j] = (np.sum(xj * rj) + alpha / 2) / zj
                        else:
                            theta_lasso[j] = 0
            if np.sum((theta_lasso - theta_prev) ** 2) < tolerance:
                break
        
        sst = np.sum((y_test - np.mean(y_test)) ** 2)

        y_pred_lasso = X_test @ theta_lasso
        mse_lasso = np.mean((y_test - y_pred_lasso) ** 2)
        ssr_lasso = np.sum((y_pred_lasso - np.mean(y_test)) ** 2)
        r_squared_lasso = 1 - (ssr_lasso / sst)

        rmse_lasso = np.sqrt(mse_lasso)

        print("Lasso regression:")
        print("Mean squared error (MSE):", mse_lasso)
        print("R-squared:", r_squared_lasso)
        print("Root mean sqaured error (RMSE):", rmse_lasso)

    def regression_pp(self):
        """
        Fits a lasso regression model on the training data using the specified regularization parameter alpha, iterations, and tolerance.
        Using processed dataset.
        """
        # Define hyperparameters
        alpha = self.alpha  # regularization strength
        max_iterations = self.max_iter  # number of gradient descent iterations
        tolerance = self.tol

        # Load the data
        diabetes_norm = pd.read_csv("diabetes_norm.csv")

        diabetes_norm.insert(0, "Intercept", 1)

        train_size = int(0.8 * len(diabetes_norm))

        X_train = diabetes_norm.iloc[:train_size, :-1].values
        y_train = diabetes_norm.iloc[:train_size, -1].values
        X_test = diabetes_norm.iloc[train_size:, :-1].values
        y_test = diabetes_norm.iloc[train_size:, -1].values

        theta_lasso = np.zeros(X_train.shape[1])
        for i in range(max_iterations):
            theta_prev = theta_lasso.copy()
            for j in range(X_train.shape[1]):
                if j == 0:
                    theta_lasso[j] = np.mean(y_train)
                else:
                    xj = X_train[:, j]
                    rj = y_train - X_train @ theta_lasso + xj * theta_lasso[j]
                    zj = xj @ xj
                    if zj == 0:
                        theta_lasso[j] = 0
                    else:
                        if np.sum(xj * rj) > alpha / 2:
                            theta_lasso[j] = (np.sum(xj * rj) - alpha / 2) / zj
                        elif np.sum(xj * rj) < - alpha / 2:
                            theta_lasso[j] = (np.sum(xj * rj) + alpha / 2) / zj
                        else:
                            theta_lasso[j] = 0
            if np.sum((theta_lasso - theta_prev) ** 2) < tolerance:
                break
        
        sst = np.sum((y_test - np.mean(y_test)) ** 2)

        y_pred_lasso = X_test @ theta_lasso
        mse_lasso = np.mean((y_test - y_pred_lasso) ** 2)
        ssr_lasso = np.sum((y_pred_lasso - np.mean(y_test)) ** 2)
        r_squared_lasso = 1 - (ssr_lasso / sst)

        rmse_lasso = np.sqrt(mse_lasso)

        print("Lasso regression:")
        print("Mean squared error (MSE):", mse_lasso)
        print("R-squared:", r_squared_lasso)
        print("Root mean sqaured error (RMSE):", rmse_lasso)