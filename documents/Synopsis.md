# PROJECT SYNOPSIS

## Fake News Detection Using Natural Language Processing and Machine Learning

---

### M.Sc. Data Science — Minor Project

---

| **Field**                | **Details**                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Project Title**        | Fake News Detection Using Natural Language Processing and Machine Learning |
| **Degree Program**       | M.Sc. Data Science                                                         |
| **Project Type**         | Minor Project                                                              |
| **Domain**               | Natural Language Processing, Machine Learning, Data Science                |
| **Programming Language** | Python 3.13                                                                |
| **Tools & Libraries**    | Pandas, NumPy, NLTK, Scikit-Learn, Matplotlib, Seaborn, WordCloud, Joblib  |

---

## 1. Introduction

The proliferation of digital media and social networking platforms has led to an unprecedented increase in the volume of information consumed daily by billions of users worldwide. While this digital revolution has democratized information access, it has simultaneously created fertile ground for the propagation of misinformation and fake news. Fake news — defined as deliberately fabricated information presented as legitimate news — poses a significant threat to democratic processes, public health, financial markets, and social cohesion.

The manual verification of news articles is an impractical approach given the sheer volume of content generated daily. This necessitates the development of automated systems capable of distinguishing between genuine and fabricated news articles with high accuracy. This project addresses this critical need by developing a **machine learning-based fake news detection system** that leverages Natural Language Processing (NLP) techniques to classify news articles as either **Real** or **Fake**.

---

## 2. Problem Statement

The rapid spread of misinformation through digital platforms has become a global concern. Traditional fact-checking mechanisms are slow, subjective, and unable to scale to the magnitude of online content. There is a critical need for automated, scalable, and accurate systems that can detect fake news in real-time.

**This project aims to develop a robust fake news detection pipeline that:**
1. Ingests and preprocesses large-scale news datasets from multiple sources.
2. Applies advanced NLP techniques for text cleaning and feature extraction.
3. Trains and evaluates multiple machine learning classifiers.
4. Identifies the best-performing model for deployment.
5. Serializes the optimal model for future production use.

---

## 3. Objectives

The primary objectives of this project are:

1. **Data Acquisition & Integration:** Collect and integrate multiple news datasets — including the global **WELFake Dataset** (25,000 samples) and the regional **Indian Fake News Dataset (IFND)** (56,714 samples) — to create a comprehensive master dataset of **81,714 articles**.

2. **Text Preprocessing:** Implement a robust NLP preprocessing pipeline incorporating:
   - Lowercasing and URL removal
   - Special character elimination using Regular Expressions
   - Stopword removal using NLTK
   - Lemmatization using WordNet Lemmatizer

3. **Exploratory Data Analysis (EDA):** Conduct thorough exploratory analysis including:
   - Class distribution analysis
   - Word count density histograms
   - Word Cloud generation for Fake and Real news corpora
   - N-gram frequency analysis (Unigrams and Bigrams)

4. **Feature Engineering:** Transform clean text data into numerical feature vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization with a maximum of 50,000 features.

5. **Model Training & Evaluation:** Train and comparatively evaluate three machine learning classifiers:
   - **Logistic Regression**
   - **Multinomial Naïve Bayes**
   - **Random Forest Classifier**

6. **Model Selection & Persistence:** Identify the best-performing model based on accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curves, and serialize it using Joblib for deployment.

---

## 4. Methodology

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology:

### Phase 1: Business Understanding
Understanding the societal impact of fake news and the need for automated detection systems.

### Phase 2: Data Understanding
- Loading the **WELFake Dataset** containing globally sourced labeled news articles.
- Loading the **IFND Dataset** containing regionally (India) sourced labeled news articles.
- Combining both datasets into a master dataset of 81,714 articles.

### Phase 3: Data Preparation
- Text cleaning using Regex, stopword removal, and lemmatization.
- Label standardization (binary: 0 = Real, 1 = Fake).
- Filtering empty or corrupt records.

### Phase 4: Modeling
- **Feature Extraction:** TF-IDF Vectorization (max 50,000 features).
- **Train-Test Split:** 80% training, 20% testing (stratified).
- **Classification Algorithms:**
  - Logistic Regression (max_iter=1000)
  - Multinomial Naïve Bayes (alpha=0.1)
  - Random Forest (n_estimators=100)

### Phase 5: Evaluation
- Accuracy Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix Visualization
- ROC-AUC Curve Comparison

### Phase 6: Deployment
- Model serialization using Joblib.
- Interactive prediction function for real-time classification.

---

## 5. Datasets Used

| **Dataset**        | **Source**     | **Records** | **Description**                                  |
|--------------------|---------------|-------------|--------------------------------------------------|
| WELFake Dataset    | Global        | 25,000      | Labeled news articles from global sources        |
| IFND Full Dataset  | India-specific| 56,714      | Labeled news articles from Indian news sources   |
| **Combined Master**| **Mixed**     | **81,714**  | **Merged dataset used for training and testing** |

---

## 6. Expected Outcomes

1. A fully functional fake news detection pipeline capable of classifying articles as Real or Fake.
2. Comparative analysis of three ML algorithms with detailed performance metrics.
3. Identification of Logistic Regression as the best-performing model.
4. A serialized model ready for deployment in production environments.
5. Comprehensive EDA visualizations revealing linguistic patterns in fake vs. real news.

---

## 7. Tools & Technologies

| **Category**            | **Technology**                                          |
|-------------------------|---------------------------------------------------------|
| Programming Language    | Python 3.13                                             |
| Data Manipulation       | Pandas, NumPy                                           |
| NLP Libraries           | NLTK (Stopwords, WordNet Lemmatizer)                    |
| Feature Extraction      | Scikit-Learn (TF-IDF Vectorizer)                        |
| ML Algorithms           | Scikit-Learn (Logistic Regression, Naive Bayes, Random Forest) |
| Visualization           | Matplotlib, Seaborn, WordCloud                          |
| Model Serialization     | Joblib                                                  |
| Development Environment | Jupyter Notebook                                        |

---

## 8. Project Timeline

| **Phase**                     | **Duration** |
|-------------------------------|--------------|
| Literature Review             | Week 1       |
| Data Collection & Preparation | Week 2     |
| EDA & Feature Engineering     | Week 4-5     |
| Model Training & Evaluation   | Week 6-7     |
| Model Optimization & Testing  | Week 8-9    |
| Report Writing & Documentation| Week 10-12   |

---

## 9. Conclusion

This project demonstrates a complete, end-to-end data science pipeline for fake news detection. By combining global and regional datasets, applying robust NLP preprocessing, and training multiple machine learning classifiers, this system achieves high accuracy in distinguishing fake news from genuine articles. The project contributes to the ongoing efforts to combat misinformation in the digital age.

---
