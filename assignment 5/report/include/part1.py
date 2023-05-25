# load the iris dataset
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
from sklearn.model_selection import train_test_split

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=None, shuffle=True, stratify=None)

import numpy as np

# Define the perceptron algorithm
class MultiClassPerceptron:
    def __init__(self, input_dim, output_dim, lr=0.01, epochs=1000):
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))
        self.lr = lr
        self.epochs = epochs

    def forward(self, X):
        weighted_sum = np.dot(X, self.W) + self.b
        probabilities = np.exp(weighted_sum) / np.sum(np.exp(weighted_sum), axis=1, keepdims=True)
        return probabilities

    def backward(self, X, y):
        m = X.shape[0]

        probabilities = self.forward(X)
        # Convert y to one-hot encoded form
        y_one_hot = np.eye(self.W.shape[1])[y]

        dW = (1 / m) * np.dot(X.T, (probabilities - y_one_hot))
        db = (1 / m) * np.sum(probabilities - y_one_hot, axis=0)

        self.W -= self.lr * dW
        self.b -= self.lr * db

    def fit(self, X, y):
        for epoch in range(self.epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        probabilities = self.forward(X)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

# Train the model
p = MultiClassPerceptron(input_dim=X_train.shape[1], output_dim=3, lr=0.0005, epochs=1000)
p.fit(X_train, y_train)
predictions_train = p.predict(X_train)
predictions = p.predict(X_test)

# evaluate train accuracy
print("Perceptron classification train accuracy", accuracy_score(y_train, predictions_train))
print("Perceptron classification accuracy", accuracy_score(y_test, predictions))