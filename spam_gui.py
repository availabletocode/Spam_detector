
# GUI APP USING TKINTER FOR SPAM NOT SPAM 
# Tkinter is built-in with Python, no need to install it separately.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# --- Load and train the model ---

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Convert text to numeric
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train-test split (we only train once, not evaluating here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# --- GUI Part ---
def check_message():
    user_input = entry.get("1.0", tk.END).strip()
    if user_input == "":
        messagebox.showwarning("Warning", "Please enter a message!")
        return
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    result = "Spam 🚫" if prediction == 1 else "Not Spam ✅"
    result_label.config(text="Result: " + result, fg="red" if prediction == 1 else "green")

# Create main window
window = tk.Tk()
window.title("Spam Detector")
window.geometry("400x300")
window.configure(bg="#f5f5f5")

# GUI Elements
title_label = tk.Label(window, text="Email / SMS Spam Detector", font=("Arial", 16, "bold"), bg="#f5f5f5")
title_label.pack(pady=10)

entry = tk.Text(window, height=5, width=40, font=("Arial", 12))
entry.pack(pady=10)

check_button = tk.Button(window, text="Check", font=("Arial", 12), command=check_message)
check_button.pack(pady=5)

result_label = tk.Label(window, text="", font=("Arial", 14), bg="#f5f5f5")
result_label.pack(pady=10)

# Run the app
window.mainloop()

# to run the code :-  python spam_gui.py