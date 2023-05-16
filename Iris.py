from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()

# Note: The Iris dataset is a well-known dataset in machine learning.
# It consists of 150 samples with measurements of iris flowers from three different species.
# The goal is to classify the species based on the sepal length, sepal width, petal length, and petal width.

# Split the dataset into features (X) and target (y)
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-nearest neighbors classifier
classifier = KNeighborsClassifier()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
