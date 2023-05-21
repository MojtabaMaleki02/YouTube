import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset from seaborn
data = sns.load_dataset("titanic")

# Preprocess the data
data.dropna(inplace=True)  # Remove rows with missing values

# Select relevant features
target = 'survived'
features = data.drop(target, axis=1) 

# Perform one-hot encoding
"""One-hot encoding is a technique used to convert categorical
variables into a binary representation that can be used by 
machine learning algorithms.
"""
data_encoded = pd.get_dummies(data, drop_first=True)

# Split the dataset into features (X) and target (y)
X = data_encoded.drop(target, axis=1)  # Features
y = data_encoded[target]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
