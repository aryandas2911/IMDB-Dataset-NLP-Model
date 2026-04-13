# IMDB Sentiment Analysis using NLP & Deep Learning

A comprehensive Natural Language Processing (NLP) project that performs sentiment analysis on the IMDB movie reviews dataset. This project explores both **classical machine learning approaches** and **deep learning (LSTM-based models)** to understand how different text representations impact performance.

---

## 📌 Project Overview

Sentiment analysis is a fundamental NLP task focused on determining whether a piece of text expresses a **positive or negative opinion**.

This project goes beyond basic implementation and focuses on:

- Comparing **traditional vectorization techniques** with **neural network-based embeddings**
- Understanding the trade-offs between **interpretability, performance, and complexity**
- Building intuition about when to use classical ML vs deep learning

---

## 🧠 Objectives

- Perform robust text preprocessing on raw movie reviews
- Experiment with multiple text vectorization techniques
- Compare classical ML models with deep learning models
- Understand strengths and limitations of each approach
- Analyze how sequence modeling improves sentiment understanding

---

## 📂 Dataset

- **Dataset:** IMDB Movie Reviews  
- **Link:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews  
- **Classes:** Positive / Negative  
- **Task:** Binary Sentiment Classification  

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python  
- **Environment:** Jupyter Notebook  

### Libraries Used

#### Classical NLP:
- NumPy  
- Pandas  
- Scikit-learn  
- Gensim  
- NLTK / Regex  

#### Deep Learning:
- TensorFlow / Keras  

---

## 🔄 Project Workflow

### 1. Text Preprocessing

- Lowercasing  
- Removing HTML tags  
- Removing special characters and numbers  
- Normalizing whitespace  

---

### 2. Train–Test Split

- Consistent split across all experiments  

---

### 3. Feature Engineering / Representation

#### 🔹 Classical NLP Techniques

- One-Hot Encoding (OHE)  
- Bag of Words (BoW)  
- TF-IDF (unigrams)  
- TF-IDF (unigrams + bigrams)  
- Word2Vec (averaged embeddings)  

#### 🔹 Deep Learning Representation

- Tokenization (Top 5000 words)  
- Sequence Padding (fixed length = 200)  
- Learned Embeddings via Keras Embedding Layer  

---

### 4. Model Training

#### Classical Model

- Logistic Regression (baseline)  

#### Deep Learning Model

- Embedding Layer  
- LSTM (Long Short-Term Memory)  
- Dense Output Layer (Sigmoid)  

---

### 5. Evaluation Metrics

- Accuracy  
- Classification Report  

---

## 📊 Experimental Results

### 🔹 Classical NLP Results

| Vectorization Method     | Accuracy |
|--------------------------|----------|
| One-Hot Encoding (OHE)   | 0.8745   |
| Bag of Words (1–2 grams) | 0.8732   |
| TF-IDF (unigrams)        | 0.8883   |
| **TF-IDF (1–2 grams)**   | **0.8920** |
| Word2Vec (Averaged)      | 0.8483   |

---

### 🔹 Deep Learning (LSTM)

| Model            | Accuracy |
|------------------|----------|
| LSTM + Embedding | 0.8910   |

---

## 🔍 Key Observations

### Classical NLP

- TF-IDF outperformed frequency-based methods by emphasizing **important words**
- Bigrams improved results by capturing **context and negation**
- Word2Vec underperformed due to **loss of word order when averaged**

### Deep Learning (LSTM)

- LSTM captures **sequential dependencies and context**
- Handles phrases like:
  > "not good" vs "good"
- Learns **word representations automatically** (no manual feature engineering)

---

## ⚖️ Classical NLP vs Deep Learning

| Aspect           | Classical NLP   | LSTM (Deep Learning) |
|------------------|-----------------|----------------------|
| Interpretability | High            | Low                  |
| Feature Control  | Manual          | Automatic            |
| Context Handling | Limited         | Strong               |
| Training Time    | Fast            | Slower               |
| Performance      | Strong baseline | Better with tuning   |

---

## ✅ Conclusion

This project shows that:

- **TF-IDF + Logistic Regression** is a strong and reliable baseline  
- **Deep learning models (LSTM)** bring contextual understanding but require more tuning  
- More complex models don’t guarantee better performance unless used correctly  

> **The best model is not the most complex one — it’s the one aligned with the problem.**