# ============================================
# Amazon Review Sentiment Prediction Script
# ============================================

import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords (first run only)
nltk.download('stopwords')

print("Loading model and vectorizer...")

# Load trained model and vectorizer
model = joblib.load("../models/sentiment_model.pkl")
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

print("Model loaded successfully.")

# ----------------------------
# Text Cleaning Function
# ----------------------------

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ----------------------------
# Load Hidden Test Data
# ----------------------------

print("Reading hidden test dataset...")

df = pd.read_csv("../data/test_data_hidden.csv")

# Combine title + text
df["review"] = df["reviews.title"].fillna("") + " " + df["reviews.text"].fillna("")

# Clean reviews
print("Cleaning reviews...")
df["clean_review"] = df["review"].apply(clean_text)

# Convert to TF-IDF
print("Transforming text to features...")
X = tfidf.transform(df["clean_review"])

# Predict sentiment
print("Predicting sentiment...")
predictions = model.predict(X)

# Add predictions to dataset
df["Predicted_Sentiment"] = predictions

# Save output
output_path = "../data/final_predictions.csv"
df.to_csv(output_path, index=False)

print("=================================")
print("Prediction completed successfully!")
print(f"Results saved to: {output_path}")
print("=================================")