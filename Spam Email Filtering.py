#for data manipulation and analysis
import pandas as pd  
# Import CountVectorizer for text preprocessing
from sklearn.feature_extraction.text import CountVectorizer  
#for splitting the dataset
from sklearn.model_selection import train_test_split  
#for the Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB  


# Load the dataset
# Load the dataset from the 'data.csv' file
# download dataset from: https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset?select=enronSpamSubset.csv
# I only used enronSpamSubset.csv file

data = pd.read_csv('data.csv')  

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['Body'], data['Label'], test_size=0.2, random_state=42)  # Split the dataset into training and testing sets

# Preprocess the email text using CountVectorizer
# Create an instance of CountVectorizer
vectorizer = CountVectorizer()  
# Fit and transform the training email text into a matrix of token counts
X_train = vectorizer.fit_transform(X_train)  
# Transform the testing email text into a matrix of token counts
X_test = vectorizer.transform(X_test)  

# Train the classifier
# Create an instance of the Multinomial Naive Bayes classifier
classifier = MultinomialNB()  
# Train the classifier using the training data
classifier.fit(X_train, y_train) 

# Evaluate the classifier
# Calculate the accuracy of the classifier on the testing data
accuracy = classifier.score(X_test, y_test)  
# Print the accuracy
print(f"Accuracy: {accuracy}")  

# Test the classifier with new email examples
new_emails = [
    "Get rich quick! Guaranteed income!",
    "Hey, let's meet for lunch today."
]
# Transform the new email examples into a matrix of token counts
new_emails_transformed = vectorizer.transform(new_emails)  
# Predict the labels (spam or not spam) for the new email examples
predictions = classifier.predict(new_emails_transformed) 
print("Predictions:")
for email, prediction in zip(new_emails, predictions):
    print(f"Email: {email}")
    print(f"Prediction: {'spam' if prediction == 1 else 'not spam'}")
    print()
