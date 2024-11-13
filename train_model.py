# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib  # For saving the model

# Load your dataset
# Replace this with your dataset (example: data.csv)
data = pd.read_csv('data.csv')

# Split the dataset into features and target variable
# Assuming the last column is the target variable
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
lr_clf = LogisticRegression(random_state=42, max_iter=1000)  # max_iter increased for convergence

# Train the model
lr_clf.fit(X_train, y_train)

# Save the model to a file
joblib.dump(lr_clf, 'logistic_regression_model.pkl')

# Make predictions
y_pred = lr_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
