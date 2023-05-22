import sklearn  # Import the scikit-learn library for machine learning
from sklearn.datasets import load_breast_cancer  # Import the breast cancer dataset
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting the data
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes classifier
from sklearn.metrics import accuracy_score  # Import accuracy_score to measure accuracy at the end


# Load the breast cancer dataset
data = load_breast_cancer()

# Retrieve the target names, labels, feature names, and features
target_names = data.target_names
labels = data.target
feature_names = data.feature_names
features = data.data

# Print the target names, label of the first sample, feature name of the first feature, and feature values of the first sample
print(target_names)
print(labels[0])
print(feature_names[0])
print(features[0])

# Split the dataset into training and testing sets
train, test, train_labels, test_labels = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
)

# Initialize and train the Gaussian Naive Bayes classifier
gnb = GaussianNB()
model = gnb.fit(train, train_labels)

# Make predictions on the test set
preds = gnb.predict(test)
print(preds)

# Print the actual labels of the test set
print(test_labels)

# Calculate and print the accuracy of the classifier
print(accuracy_score(test_labels, preds))
