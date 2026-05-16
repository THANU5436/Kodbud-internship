import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'message': [
        'Win money now',
        'Hello how are you',
        'Claim your free prize',
        'Meeting at 5 PM',
        'Congratulations you won lottery'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)

# Features and labels
X = df['message']
y = df['label']

# Convert text to vectors
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict new message
new_message = ["You have won a free iPhone"]
new_vector = vectorizer.transform(new_message)

prediction = model.predict(new_vector)

print("Prediction:", prediction[0])