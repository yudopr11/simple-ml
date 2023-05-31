import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calculate the regression coefficients using the normal equation
        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Make predictions
        predictions = X @ self.coefficients
        return predictions

# Usage example
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 5, 4, 6])

X_test = np.array([[6], [7], [8]])
y_test = np.array([7, 8, 8])

# Create a linear regression model object
regressor = LinearRegression()

# Fit the model to the training data
regressor.fit(X_train, y_train)

# Get the coefficients
coefficients = regressor.coefficients

# Make predictions on test data
y_pred = regressor.predict(X_test)

# Visualization of prediction results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
