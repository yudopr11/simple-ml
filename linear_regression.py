import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# read data
data = pd.read_csv('data.csv')

# separate independent and dependent variables
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# split the data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# create a linear regression model object
regressor = LinearRegression()

# fit the model to the training data
regressor.fit(X_train, y_train)

# get the coefficients
coefficients = regressor.coefficients

# make predictions on new data
y_pred = regressor.predict(X_test)

# visualization of prediction results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Gaji vs. Tahun Pengalaman (Data Uji)')
plt.xlabel('Tahun Pengalaman')
plt.ylabel('Gaji')
plt.show()
