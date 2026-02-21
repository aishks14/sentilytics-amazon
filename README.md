# Amazon Product Review Sentiment Analysis (E-commerce NLP Project)

## Project Overview
This project performs sentiment analysis on Amazon product reviews in the e-commerce domain.  
The system automatically classifies customer reviews into:

- Positive
- Neutral
- Negative

In addition to sentiment prediction, the project also extracts hidden discussion topics from customer reviews to understand *why* customers are satisfied or dissatisfied.

The project demonstrates an end-to-end Natural Language Processing (NLP) pipeline including:
- Exploratory Data Analysis
- Class imbalance handling
- Machine learning modeling
- Deep learning modeling
- Topic modeling
- Model deployment

---

## Project Structure

```
sentilytics-amazon/
│
├── data/
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── test_data_hidden.csv
│
├── notebooks/
│   ├── 01_EDA_Imbalance.ipynb
│   ├── 02_TFIDF_ML_Models.ipynb
│   ├── 03_DeepLearning_LSTM.ipynb
│   ├── 04_Topic_Modeling.ipynb
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── sentiment_model.pkl
│   ├── lstm_model.h5
│
├── src/
│   ├── predict.py
│
├── requirements.txt
└── README.md
```


---

## Dataset Information

The dataset contains Amazon product reviews with the following attributes:

- Brand
- Category
- Review Title
- Review Text
- Sentiment label (Positive, Neutral, Negative)

Files used:
- `train_data.csv` → used for training the models
- `test_data.csv` → used for validation
- `test_data_hidden.csv` → used for final prediction

---

## Installation

### Step 1: Install Python
Install **Python 3.10 or higher**.

Check version:
`python --version`


---

### Step 2: Install Required Libraries
Navigate to the project root folder and run:
`pip install -r requirements.txt`


This will automatically install all required packages.

---


---

## How to Reproduce the Project

You must run notebooks in the **exact order below**.

### Step 1 — Data Analysis & Preprocessing
Open and run:

`notebooks/01_EDA_Imbalance.ipynb`


This will:
- Perform EDA
- Clean reviews
- Handle class imbalance
- Create `data/processed_train.csv`

---

### Step 2 — Machine Learning Models
Run:
`notebooks/02_TFIDF_ML_Models.ipynb`

This will:
- Train Naive Bayes and SVM
- Evaluate performance
- Save trained model

Outputs created:

`models/sentiment_model.pkl`
`models/tfidf_vectorizer.pkl`

---

### Step 3 — Deep Learning Model
Run:
`notebooks/03_DeepLearning_LSTM.ipynb`

This will:
- Train LSTM neural network
- Compare with ML models
- Save neural network

Outputs:
`models/lstm_model.keras`
`models/tokenizer.pkl`

---

### Step 4 — Topic Modeling
Run:
`notebooks/04_Topic_Modeling.ipynb`

This will:
- Discover hidden topics
- Generate review themes
- Save topic dataset

Output:
`data/review_topics.csv`

---

## Running Final Prediction

After models are created, generate predictions for unseen reviews.

Open terminal:
`python predict.py`

This script will:

1. Load the trained SVM model
2. Clean hidden test reviews
3. Convert text into TF-IDF features
4. Predict sentiment

Output file generated:
`data/final_predictions.csv`

---

## Model Selection
Multiple models were tested:

| Model | Result |
|------|------|
| Naive Bayes | Good baseline |
| SVM | Best performance |
| LSTM | Lower accuracy due to dataset size |

The **SVM classifier** was selected as the final deployment model because it achieved the highest accuracy and stability.

---

## Business Insights (Topic Modeling)
Topic modeling revealed that customer feedback mainly focuses on:

- Battery performance
- Delivery service
- Packaging quality
- Product durability
- Display and sound quality

This helps companies improve product and service quality.

---

## Conclusion
This project demonstrates a complete end-to-end NLP pipeline for customer feedback analytics in e-commerce.  
It combines machine learning, deep learning, and topic modeling to both predict sentiment and understand customer concerns.

The system can automatically analyze thousands of reviews and provide actionable insights to businesses.

### Important Note

If NLTK stopwords are not available, run the following once inside Python:

```python
import nltk
nltk.download('stopwords')
```

## Author
Aishwarya Kumar Singh <br>
Data Engineering & Analytics
