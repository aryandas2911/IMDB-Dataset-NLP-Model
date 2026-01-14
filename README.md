# IMDB Sentiment Analysis using NLP

A notebook-based Natural Language Processing (NLP) project that performs sentiment analysis on the IMDB movie reviews dataset. The project focuses on comparing different text vectorization techniques and understanding how they affect classification performance.

---

## 📌 Project Overview

Sentiment analysis is a core NLP task where the goal is to determine whether a piece of text expresses a positive or negative opinion. In this project, various text representation techniques are applied to IMDB movie reviews and evaluated using a classical machine learning classifier.

Rather than building an end-to-end application, this project emphasizes **conceptual clarity, experimentation, and interpretation of results**, making it ideal as a first NLP project.

---

## 🧠 Objectives

- Perform thorough text preprocessing on raw movie reviews  
- Experiment with multiple text vectorization techniques  
- Compare model performance under identical conditions  
- Understand when and why certain representations work better  
- Draw meaningful conclusions from experimental results  

---

## 📂 Dataset

- **Dataset:** IMDB Movie Reviews
- **Link:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Classes:** Positive / Negative  
- **Task:** Binary sentiment classification  

Each review is labeled based on the sentiment expressed by the reviewer.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python  
- **Environment:** Jupyter Notebook  

**Libraries Used:**
- NumPy  
- Pandas  
- Scikit-learn  
- Gensim  
- NLTK / Regex (for preprocessing)

---

## 🔄 Project Workflow

The project follows a structured NLP pipeline:

1. **Text Preprocessing**
   - Lowercasing
   - Removing HTML tags
   - Removing special characters and numbers
   - Normalizing extra whitespace

2. **Train–Test Split**
   - Single split applied consistently across all experiments

3. **Text Vectorization**
   - One-Hot Encoding (OHE)
   - Bag of Words (BoW)
   - TF-IDF (unigrams)
   - TF-IDF (unigrams + bigrams)
   - Word2Vec (averaged word embeddings)

4. **Model Training**
   - Logistic Regression used as a baseline classifier

5. **Evaluation**
   - Accuracy
   - Classification report

---

## 📊 Experimental Results

| Vectorization Method        | Accuracy |
|-----------------------------|----------|
| One-Hot Encoding (OHE)      | 0.8745   |
| Bag of Words (1–2 grams)    | 0.8732   |
| TF-IDF (unigrams)           | 0.8883   |
| **TF-IDF (1–2 grams)**      | **0.8920** |
| Word2Vec (Averaged Vectors) | 0.8483   |

---

## 🔍 Key Observations

- TF-IDF outperformed frequency-based methods by emphasizing sentiment-bearing words.
- Incorporating bigrams improved performance by capturing contextual patterns such as negation.
- Word2Vec embeddings, while semantically rich, underperformed due to loss of word order and negation when using simple averaging.
- More complex representations do not necessarily yield better results without appropriate modeling techniques.

---

## ✅ Conclusion

This project demonstrates that the choice of text representation plays a crucial role in NLP pipelines. For sentiment analysis tasks that rely heavily on specific keywords and short contextual patterns, TF-IDF with n-grams can outperform semantic embeddings like Word2Vec.

The project highlights the importance of aligning model complexity with task requirements rather than assuming that more advanced techniques will always lead to better performance.

## 📎 Notes

This project is intended for learning and academic purposes and focuses on building a strong conceptual foundation in NLP rather than production deployment.

---
