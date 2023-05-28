from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
"""
Mean Squared Error (MSE): Mean Squared Error measures the average
squared difference between the predicted values (y_pred) and
the actual values (y_test). It provides a measure of how close the predicted
values are to the actual values. 
A lower MSE indicates better accuracy and performance of the model.

R-squared (R2) Score: R-squared is a statistical measure that represents the proportion
of the variance in the dependent variable (y) that can be explained by the independent 
variables (X) in the linear regression model. R2 score ranges from 0 to 1, where 1 indicates
a perfect fit and 0 indicates no linear relationship between the variables. It provides
an indication of how well the model fits the data. Higher R2 scores indicate a better fit.
"""
# Load the Boston Housing dataset
data = load_boston()

# Split the dataset into features (X) and target (y)
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)
