class Predictor:
    def __init__(self, dataset):
        self.coefficients = None
        self.df = pd.read_csv(dataset)
        self.df_pp = self.preprocess(self.df)
        self.X_train_raw, self.X_test_raw, self.y_train_raw, self.y_test_raw = self.train_test_split(self.df)
        self.X_train_pp, self.X_test_pp, self.y_train_pp, self.y_test_pp = self.train_test_split(self.df_pp)

    def preprocess(self, df):
        # It is better to do this after splitting the dataset. --> need to change that
        # Handle missing values
        df.replace(0, np.nan, inplace=True)

        # Remove outliers using iqr rule:
        df_cleaned = df.copy()
        for column in df:
            q1 = df[column].quantile(q=0.25)
            q3 = df[column].quantile(q=0.75)
            med = df[column].median()

            iqr = q3 - q1
            upper_bound = q3+(1.5*iqr)
            lower_bound = q1-(1.5*iqr)

            df_cleaned[column][(df[column] <= lower_bound) | (df[column] >= upper_bound)] = df_cleaned[column].median()

        # Normalize data:
        df_normalized = df_cleaned.copy()
        for column in df_cleaned:

            mean = df_cleaned[column].mean()
            std = df_cleaned[column].std()

            df_normalized[column] = (df_cleaned[column] - mean) / std
        df_processed = df_normalized
        return df_processed

    def train_test_split(self, df, test_size=0.2):
        # Shuffle the rows of the dataset randomly
        df_randomized = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Extract the features and target variable
        X = df_randomized.drop('target', axis=1)
        y = df_randomized['target']

        # Split the dataset into training and testing sets
        split_ratio = 1 - test_size
        split_index = int(split_ratio * len(df_randomized))

        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]

        return X_train, X_test, y_train, y_test
        
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass