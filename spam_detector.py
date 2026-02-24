# trained in msg spm/not spam data set from kaggle for more realistic email/ msg we can use differnt dataset 

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Step 2: Convert labels (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 3: Feature extraction - Convert text to numeric
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 5: Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Step 7: Test with custom message
while True:
    msg = input("\nEnter a message to test (or type 'exit'): ")
    if msg.lower() == "exit":
        break
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    print("Prediction:", "Spam" if prediction == 1 else "Not Spam")
