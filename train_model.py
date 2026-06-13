import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Create model folder
os.makedirs("model", exist_ok=True)


# Load dataset
data = pd.read_csv("dataset/resume_dataset.csv")


print("Dataset Loaded")
print(data.head())


# Remove empty values

data = data.dropna()


# Input and output

X = data["Resume"]
y = data["Category"]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Convert text into numbers

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)


X_train = tfidf.fit_transform(X_train)

X_test = tfidf.transform(X_test)



# Train Machine Learning Model

model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train,
    y_train
)



# Prediction

prediction = model.predict(X_test)



# Accuracy

accuracy = accuracy_score(
    y_test,
    prediction
)


print("Model Accuracy:", accuracy)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        prediction
    )
)



# Save model

pickle.dump(
    model,
    open("model/resume_model.pkl","wb")
)


pickle.dump(
    tfidf,
    open("model/tfidf_vectorizer.pkl","wb")
)


print("\nModel saved successfully")