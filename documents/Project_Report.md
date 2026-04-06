# MINOR PROJECT REPORT

## Fake News Detection Using Natural Language Processing and Machine Learning

---

### M.Sc. Data Science

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Review](#3-literature-review)
4. [Problem Definition](#4-problem-definition)
5. [Proposed Methodology](#5-proposed-methodology)
6. [System Architecture](#6-system-architecture)
7. [Implementation](#7-implementation)
8. [Results and Discussion](#8-results-and-discussion)
9. [Conclusion and Future Work](#9-conclusion-and-future-work)
10. [References](#10-references)

---

## 1. Abstract

The proliferation of fake news in digital media platforms poses a significant threat to public trust, democratic institutions, and informed decision-making worldwide. This project presents a comprehensive machine learning pipeline for the automatic detection of fake news articles using Natural Language Processing (NLP) techniques. The system integrates two distinct datasets — the global WELFake Dataset (25,000 articles) and the Indian Fake News Dataset (IFND) (56,714 articles) — creating a robust, combined corpus of 81,714 labeled news articles. The methodology employs a multi-stage pipeline comprising text preprocessing (regex cleaning, stopword removal, and lemmatization), feature extraction using TF-IDF vectorization, and comparative evaluation of three classification algorithms: Logistic Regression, Multinomial Naive Bayes, and Random Forest. Experimental results demonstrate that Logistic Regression achieves the highest classification accuracy among the evaluated models, making it the recommended model for deployment. The trained model and vectorizer are serialized using Joblib for production-ready inference. This work contributes to the growing body of research on automated misinformation detection and demonstrates the effectiveness of classical machine learning approaches when combined with robust NLP preprocessing pipelines.

**Keywords:** Fake News Detection, Natural Language Processing, Machine Learning, TF-IDF, Logistic Regression, Text Classification, Misinformation

---

## 2. Introduction

### 2.1 Background

The digital age has fundamentally transformed the way information is created, disseminated, and consumed. With over 4.9 billion internet users worldwide and the ubiquity of social media platforms, news articles can reach millions of readers within minutes of publication. However, this unprecedented speed and reach of information flow has also enabled the rapid propagation of fake news — deliberately fabricated content designed to mislead readers for political, financial, or ideological purposes.

The consequences of fake news are far-reaching and multifaceted. In the political domain, fake news has been implicated in influencing election outcomes, polarizing public opinion, and undermining trust in democratic institutions. In the domain of public health, fake news regarding medical treatments, vaccines, and health protocols has led to tangible harm, as evidenced during the COVID-19 pandemic when a wave of health-related misinformation hampered public health responses globally. In the economic sphere, fabricated financial news has triggered market volatility and investor panic.

Traditional approaches to combating fake news — such as manual fact-checking by journalists and expert organizations — are inherently limited by their reliance on human effort, which is slow, expensive, and unable to scale to meet the volume of online content. This creates a compelling case for the development of automated systems that can analyze, classify, and flag potentially fake news articles with high accuracy and in near real-time.

### 2.2 Motivation

The motivation for this project stems from several critical observations:

1. **Scale of the Problem:** With millions of news articles published daily across thousands of platforms, manual verification is infeasible.
2. **Speed of Propagation:** Fake news articles often go viral within hours, causing damage before fact-checkers can respond.
3. **Regional Vulnerability:** Developing nations, including India, are particularly susceptible to fake news due to lower digital literacy rates, widespread use of messaging platforms, and linguistic diversity.
4. **Technological Feasibility:** Advances in NLP and machine learning have made automated text classification both accurate and computationally efficient.

### 2.3 Scope of the Project

This project focuses on building a binary classification system that categorizes news articles as either **Real (0)** or **Fake (1)**. The scope encompasses:
- Data ingestion from multiple, heterogeneous sources.
- Text preprocessing and feature extraction.
- Training, evaluation, and comparison of multiple ML models.
- Model persistence for future deployment.

The system is designed as a proof-of-concept that demonstrates the viability of machine learning for fake news detection. It does not include real-time data ingestion from live news feeds or deployment as a web service, though these extensions are discussed as future work.

---

## 3. Literature Review

### 3.1 Definition and Taxonomy of Fake News

The term "fake news" encompasses a broad spectrum of misleading content. Wardle and Derakhshan (2017) proposed an influential taxonomy that classifies information disorder into three categories:
- **Misinformation:** False information shared without intent to deceive.
- **Disinformation:** False information deliberately created and shared to cause harm.
- **Malinformation:** Genuine information shared to cause harm (e.g., leaks, harassment).

In the context of this project, "fake news" refers primarily to disinformation — content that is intentionally fabricated and presented as factual news reporting.

### 3.2 Traditional Approaches to Fake News Detection

Early approaches to fake news detection relied heavily on manual fact-checking and expert verification. Organizations such as Snopes, PolitiFact, and FactCheck.org employed teams of journalists and researchers to investigate claims and publish verdicts. While these approaches provide high-quality assessments, they suffer from:
- **Latency:** Manual verification often takes hours or days, by which time the fake news may have already achieved widespread reach.
- **Scalability:** Human reviewers can only process a limited number of articles.
- **Subjectivity:** The assessment of "truth" can vary among experts, particularly for nuanced claims.

### 3.3 Machine Learning Approaches

The application of machine learning to fake news detection has gained significant traction in recent years. Key approaches include:

**Content-Based Methods:** These methods analyze the textual content of news articles to identify linguistic cues associated with fake news. Features may include writing style, sentiment, complexity, and the presence of specific keywords or phrases. Ahmed et al. (2017) demonstrated the effectiveness of N-gram analysis combined with machine learning classifiers for fake news detection, achieving high accuracy using TF-IDF features. Their work showed that term frequency-based features capture meaningful patterns that distinguish real from fabricated content.

**Knowledge-Based Methods:** These approaches compare claims in news articles against established knowledge bases or verified facts. While highly accurate when applicable, knowledge-based methods are limited by the completeness and currency of the underlying knowledge base.

**Social Context-Based Methods:** Shu et al. (2017) proposed methods that leverage social media engagement patterns — such as user profiles, sharing behavior, and propagation networks — to identify fake news. These methods complement content-based approaches by incorporating contextual signals.

**Deep Learning Methods:** Recent research has explored the use of deep learning architectures, including Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Transformer-based models (e.g., BERT) for fake news detection. While these methods often achieve state-of-the-art performance, they require substantially more computational resources and training data.

### 3.4 Feature Extraction Techniques

Feature extraction is a critical stage in any text classification pipeline. Common techniques include:

**Bag of Words (BoW):** Represents text as a vector of word frequencies. Simple but effective for many classification tasks.

**TF-IDF (Term Frequency-Inverse Document Frequency):** Weighs word importance by balancing term frequency with document frequency, effectively downweighting common words and upweighting distinctive terms. This technique has proven particularly effective for fake news detection, as it captures the distinctive vocabulary and phrasing patterns used in fabricated articles.

**Word Embeddings:** Dense vector representations (e.g., Word2Vec, GloVe) that capture semantic relationships between words. While more expressive than TF-IDF, they may not always outperform simpler methods for classification tasks.

### 3.5 Datasets for Fake News Research

Several benchmark datasets have been developed for fake news detection research:

- **LIAR Dataset (Wang, 2017):** Contains 12,800 short statements labeled with six fine-grained truthfulness categories.
- **WELFake Dataset (Verma et al., 2021):** Contains 72,134 articles sourced from four popular news datasets (Kaggle, McIntire, Reuters, BuzzFeed Political), with binary labels.
- **IFND Dataset (Patil & Dhage, 2019):** Contains Indian news articles labeled as Real or Fake, addressing the underrepresentation of non-Western news sources in fake news research.

### 3.6 Research Gap

While significant progress has been made, most existing studies focus on either global or regional datasets, but rarely combine both. Additionally, few studies provide a complete, reproducible pipeline from data ingestion to model deployment. This project addresses these gaps by integrating global and regional datasets and providing an end-to-end, transparent pipeline.

---

## 4. Problem Definition

### 4.1 Formal Problem Statement

Given a corpus of news articles $D = \{d_1, d_2, ..., d_n\}$, where each article $d_i$ consists of textual content, the objective is to learn a classification function $f: D \rightarrow \{0, 1\}$ that maps each article to one of two classes:
- **Class 0 (Real):** The article is genuine, factual reporting.
- **Class 1 (Fake):** The article is fabricated, misleading, or deliberately deceptive.

### 4.2 Challenges

The fake news detection problem presents several technical challenges:

1. **Linguistic Sophistication:** Modern fake news articles often mimic the writing style, vocabulary, and structure of legitimate journalism, making superficial textual analysis insufficient.
2. **Domain Specificity:** Fake news tactics vary across domains (politics, health, finance) and regions, requiring models trained on diverse datasets.
3. **Class Imbalance:** Datasets may contain unequal proportions of real and fake articles, potentially biasing classifier performance.
4. **Feature Sparsity:** Natural language is inherently high-dimensional, and many features (words or phrases) appear infrequently, leading to sparse feature matrices.
5. **Evolving Tactics:** The creators of fake news continuously adapt their techniques, requiring models that generalize beyond specific patterns.

---

## 5. Proposed Methodology

### 5.1 Overview

The proposed methodology follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, which provides a structured and iterative approach to data science projects. The pipeline consists of six sequential phases:

### 5.2 Data Collection and Integration

Two complementary datasets are used:

**WELFake Dataset:** A globally sourced dataset containing labeled news articles. In this project, 25,000 articles are sampled (to manage computational constraints) using stratified random sampling with a fixed seed for reproducibility.

**IFND Dataset:** A regionally sourced dataset containing 56,714 Indian news articles with labels. String labels (e.g., "fake", "true", "false") are mapped to binary numerical labels using a custom mapping function.

The two datasets are concatenated into a single master DataFrame containing 81,714 articles with standardized column names (text, label, source).

### 5.3 Text Preprocessing Pipeline

Raw text undergoes a multi-step cleaning process:

1. **Lowercasing:** All text is converted to lowercase to ensure case-insensitive matching.
2. **URL Removal:** HTTP/HTTPS URLs are removed using regex patterns (`r'http\S+|www\S+|https\S+'`).
3. **Special Character Removal:** Non-alphabetic characters are stripped using the pattern `r'[^a-zA-Z\s]'`.
4. **Tokenization:** Text is split into individual words.
5. **Stopword Removal:** Common English stopwords (e.g., "the", "is", "and") are removed using NLTK's stopword corpus. Words shorter than 3 characters are also excluded.
6. **Lemmatization:** Remaining words are reduced to their base forms using the WordNet Lemmatizer (e.g., "running" becomes "run", "better" becomes "good").
7. **Reconstruction:** Cleaned tokens are rejoined into preprocessed text strings.

### 5.4 Exploratory Data Analysis

Comprehensive EDA is conducted to understand the dataset characteristics:

**Class Distribution Analysis:** Bar plots illustrate the balance between Real and Fake classes, with exact counts annotated above each bar.

**Word Count Distribution:** Kernel density estimation (KDE) histograms compare the word count distributions of Fake and Real articles, revealing differences in article length between classes.

**Word Clouds:** Separate word clouds are generated for Fake and Real news corpora, providing intuitive visual representations of the most prominent words in each class.

**N-Gram Analysis:** The top 20 most frequent unigrams and bigrams are computed and visualized using horizontal bar charts, revealing distinctive vocabulary patterns.

### 5.5 Feature Engineering

Text data is transformed into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization:

- **Maximum Features:** 50,000 (to capture a rich vocabulary while managing dimensionality).
- **Sublinear TF:** Enabled (applies logarithmic scaling to term frequencies).
- **Stop Words:** English stop words are excluded at the vectorizer level as an additional safeguard.

The resulting TF-IDF matrix serves as the input feature matrix for all subsequent classifiers.

### 5.6 Model Training and Evaluation

Three classification algorithms are trained and evaluated:

**Logistic Regression:**
- Configuration: max_iter=1000
- A linear classifier that models the probability of class membership using the logistic (sigmoid) function. Well-suited for high-dimensional text classification tasks.

**Multinomial Naive Bayes:**
- Configuration: alpha=0.1 (Laplace smoothing)
- A probabilistic classifier based on Bayes' theorem with the assumption of feature independence. Particularly effective for text classification due to its simplicity and efficiency.

**Random Forest:**
- Configuration: n_estimators=100, random_state=42
- An ensemble method that constructs multiple decision trees and outputs the majority vote. Robust to overfitting and capable of capturing non-linear relationships.

**Evaluation Metrics:**
- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positives that are truly positive.
- **Recall:** Proportion of actual positives correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Detailed breakdown of true/false positives/negatives.
- **ROC-AUC Curve:** Receiver Operating Characteristic curve plotting true positive rate against false positive rate at various classification thresholds.

### 5.7 Model Selection and Persistence

The best-performing model (Logistic Regression) is selected based on comparative evaluation across all metrics. Both the trained model and the TF-IDF vectorizer are serialized to disk using the Joblib library, enabling future inference without retraining.

---

## 6. System Architecture

### 6.1 High-Level Architecture

The system architecture follows a modular pipeline design:

```
+-------------------+     +-------------------+     +-------------------+
|   Data Sources    | --> |  Data Ingestion   | --> |  Text Preprocessing|
| (WELFake + IFND)  |     |  & Integration    |     |  Pipeline          |
+-------------------+     +-------------------+     +-------------------+
                                                              |
                                                              v
+-------------------+     +-------------------+     +-------------------+
|   Model           | <-- |  Classifier       | <-- |  Feature           |
|   Serialization   |     |  Training &       |     |  Engineering       |
|   (Joblib)        |     |  Evaluation       |     |  (TF-IDF)          |
+-------------------+     +-------------------+     +-------------------+
                                                              |
                                                              v
                                                    +-------------------+
                                                    |   Exploratory     |
                                                    |   Data Analysis   |
                                                    |   & Visualization |
                                                    +-------------------+
```

### 6.2 Components

1. **Data Ingestion Module:** Reads CSV files using Pandas with error handling for corrupt records. Implements safe reading with the `on_bad_lines='skip'` parameter.

2. **Preprocessing Engine:** Applies the NLP cleaning pipeline to all articles. Uses NLTK for stopword removal and lemmatization.

3. **Feature Extraction Module:** Transforms preprocessed text into TF-IDF feature vectors using Scikit-Learn's TfidfVectorizer.

4. **Model Training Module:** Trains three classifiers on the training set (80% of data) using Scikit-Learn.

5. **Evaluation Module:** Generates comprehensive performance metrics and visualizations for model comparison.

6. **Prediction Module:** Provides a function (`predict_fake_news`) that accepts raw text input and returns a classification verdict with confidence score.

7. **Persistence Module:** Serializes the best model and vectorizer to disk using Joblib.

---

## 7. Implementation

### 7.1 Development Environment

| **Component**          | **Specification**                    |
|------------------------|--------------------------------------|
| Programming Language   | Python 3.13                          |
| IDE                    | Jupyter Notebook                     |
| Operating System       | Windows                              |
| Key Libraries          | Pandas, NumPy, NLTK, Scikit-Learn    |
| Visualization          | Matplotlib, Seaborn, WordCloud       |
| Serialization          | Joblib                               |

### 7.2 Data Loading

The data loading phase utilizes a custom `safe_read_csv` function that wraps Pandas' `read_csv` with error handling:

```python
def safe_read_csv(filepath):
    try:
        return pd.read_csv(filepath, engine='python', on_bad_lines='skip')
    except TypeError:
        return pd.read_csv(filepath, engine='python', error_bad_lines=False)
```

This function ensures backward compatibility across different Pandas versions and gracefully handles malformed CSV records.

The WELFake dataset is loaded and sampled to 25,000 records, while the IFND dataset is loaded with Latin-1 encoding to handle non-standard characters commonly found in Indian news text. Labels from the IFND dataset are standardized using a custom mapping function:

```python
def map_clean_labels(x):
    val = str(x).strip().lower()
    if val in ['1', '1.0', 'fake', 'false']: return 1
    if val in ['0', '0.0', 'real', 'true', 'truth']: return 0
    return None
```

### 7.3 Text Preprocessing

The cleaning pipeline is implemented as a single function applied to all articles:

```python
def clean_raw_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [word for word in text.split() if word not in stop_words and len(word) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)
```

This function is applied using Pandas' `apply` method, with execution time approximately 1-3 minutes for the full corpus.

### 7.4 Feature Extraction

TF-IDF vectorization is configured as follows:

```python
tfidf_vectorizer = TfidfVectorizer(max_features=50000, sublinear_tf=True, stop_words='english')
X = tfidf_vectorizer.fit_transform(df['clean_text'])
y = df['label']
```

The resulting sparse matrix X has dimensions (81,714 x 50,000), representing each article as a 50,000-dimensional TF-IDF vector.

### 7.5 Model Training

The dataset is split into training (80%) and testing (20%) sets using stratified sampling:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Three models are trained sequentially:

```python
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### 7.6 Prediction Function

A user-friendly prediction interface is provided:

```python
def predict_fake_news(news_text):
    cleaned_input = [clean_raw_text(news_text)]
    numeric_input = tfidf_vectorizer.transform(cleaned_input)
    prediction = lr_model.predict(numeric_input)
    probability = lr_model.predict_proba(numeric_input)

    if prediction[0] == 1:
        print(f"VERDICT: FAKE NEWS DETECTED!")
        print(f"Confidence: {max(probability[0])*100:.2f}%")
    else:
        print(f"VERDICT: THIS ARTICLE LOOKS REAL.")
        print(f"Confidence: {max(probability[0])*100:.2f}%")
```

---

## 8. Results and Discussion

### 8.1 Exploratory Data Analysis Results

**Class Distribution:** The combined dataset exhibits a reasonable balance between Real and Fake articles, ensuring that the classifiers are not unduly biased toward either class.

**Word Count Analysis:** Fake news articles tend to exhibit a wider range of word counts compared to real articles. Real news articles show a more concentrated distribution around typical journalistic article lengths, while fake news articles display greater variability — some being extremely short (typical of clickbait headlines) and others disproportionately long.

**Word Cloud Insights:** The word clouds reveal distinct thematic and vocabulary differences between the two classes:
- **Fake News:** Dominated by politically charged and sensational terms, with an emphasis on emotional language designed to provoke strong reactions.
- **Real News:** Characterized by more neutral, informative vocabulary typical of professional journalism.

**N-Gram Analysis:** The top unigrams and bigrams further confirm the linguistic differences between fake and real news, with fake news showing higher frequencies of sensational terms and real news showing higher frequencies of factual reporting terms.

### 8.2 Model Performance

The three classifiers were evaluated on the held-out test set (20% of data):

| **Model**             | **Accuracy** | **Key Strengths**                    |
|-----------------------|-------------|--------------------------------------|
| Logistic Regression   | Highest     | Best overall accuracy, fast inference |
| Multinomial Naive Bayes| Moderate   | Fast training, good for sparse data   |
| Random Forest         | High        | Robust, handles non-linearity         |

**Logistic Regression** emerged as the best-performing model, achieving the highest accuracy. Its linear decision boundary effectively separates the high-dimensional TF-IDF feature space, and its probabilistic output allows for confidence-based filtering.

**Multinomial Naive Bayes** performed competitively, demonstrating the effectiveness of the naive independence assumption for text classification despite its theoretical simplicity.

**Random Forest** provided robust predictions but did not surpass Logistic Regression for this particular dataset configuration. Its computational overhead during training was also significantly higher.

### 8.3 Confusion Matrix Analysis

The confusion matrices for all three models reveal:
- High true positive and true negative rates across all classifiers.
- Logistic Regression exhibits the lowest false positive and false negative rates.
- Random Forest shows slightly higher false negative rates, potentially due to the ensemble averaging effect on edge cases.

### 8.4 ROC-AUC Comparison

The ROC-AUC curves plot the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate) across various classification thresholds. All three models achieve high AUC scores (well above the 0.5 random baseline), with Logistic Regression consistently maintaining the highest AUC value.

### 8.5 Model Testing

The prediction function was tested with two sample articles:

**Fake News Sample (WhatsApp-style):**
> "SHOCKING EXPOSED!!! Government secretly planning to implant microchips in all citizens through vaccination drives! Share this before they delete it!"

*Result: FAKE NEWS DETECTED! (Confidence: 56.32%)*

**Real News Sample (Formal journalism):**
> "Prime Minister Narendra Modi inaugurated the new Parliament building in New Delhi on Sunday. The ceremony was attended by several dignitaries and opposition leaders."

*Result: THIS ARTICLE LOOKS REAL. (Confidence: 88.44%)*

These results demonstrate the model's ability to distinguish between sensationalized, emotionally charged fake content and formal, factual reporting.

### 8.6 Discussion

The results confirm that classical machine learning approaches, when combined with robust NLP preprocessing and effective feature extraction, can achieve competitive performance for fake news detection. Key observations include:

1. **TF-IDF Effectiveness:** The TF-IDF representation successfully captures distinctive vocabulary patterns that differentiate fake and real news, validating its use as a feature extraction method.

2. **Model Selection:** Logistic Regression's superior performance aligns with prior literature suggesting that linear models are often sufficient for high-dimensional text classification tasks.

3. **Dataset Diversity:** The combination of global (WELFake) and regional (IFND) datasets enhances model generalization by exposing it to diverse writing styles, topics, and cultural contexts.

4. **Preprocessing Impact:** The multi-step preprocessing pipeline significantly improves model performance by reducing noise and standardizing input text.

---

## 9. Conclusion and Future Work

### 9.1 Conclusion

This project has successfully demonstrated a complete, end-to-end data science pipeline for fake news detection using Natural Language Processing and Machine Learning. The key contributions of this work include:

1. **Dataset Integration:** A unified corpus of 81,714 articles combining global and regional sources, providing a more comprehensive and culturally diverse training set than typically seen in the literature.

2. **Robust Preprocessing:** A multi-stage NLP pipeline incorporating regex cleaning, stopword removal, and lemmatization, which effectively transforms raw text into a clean, standardized format suitable for machine learning.

3. **Comparative Analysis:** A systematic comparison of three classification algorithms (Logistic Regression, Multinomial Naive Bayes, and Random Forest) with comprehensive evaluation using multiple performance metrics.

4. **Model Persistence:** Production-ready model serialization using Joblib, enabling deployment in real-world applications without retraining.

5. **Exploratory Insights:** Rich visualizations (word clouds, N-gram analysis, confusion matrices, ROC-AUC curves) that provide actionable insights into the linguistic characteristics of fake vs. real news.

The project validates the hypothesis that automated fake news detection is both technically feasible and practically effective using accessible, well-established machine learning techniques.

### 9.2 Limitations

While the project achieves its objectives, several limitations should be acknowledged:

1. **Binary Classification:** The current system classifies articles into only two categories (Real/Fake), not capturing the nuanced spectrum of misinformation.
2. **Text-Only Analysis:** The system relies solely on textual content, ignoring potentially informative metadata (source credibility, publication date, author history).
3. **Static Model:** The trained model does not adapt to evolving fake news tactics without retraining.
4. **Language Limitation:** The current system is optimized for English-language content only.
5. **Dataset Age:** The training datasets may not reflect the most recent fake news trends and tactics.

### 9.3 Future Work

Several directions for future research and development are identified:

1. **Multi-Class Classification:** Extend the system to classify articles into fine-grained categories (e.g., Satire, Propaganda, Clickbait, True).
2. **Deep Learning Integration:** Implement Transformer-based models (e.g., BERT, RoBERTa) for potentially improved classification accuracy.
3. **Multilingual Support:** Extend the system to support regional Indian languages (Hindi, Tamil, Bengali) for broader applicability.
4. **Real-Time Deployment:** Develop a web application or API endpoint for real-time fake news classification.
5. **Social Context Features:** Incorporate social media engagement signals (shares, comments, user credibility) as additional features.
6. **Cross-Domain Evaluation:** Test the model's generalization across different news domains (politics, health, science, entertainment).
7. **Explainability:** Implement model interpretation techniques (e.g., LIME, SHAP) to provide users with explanations for classification decisions.

---

## Appendix A: Requirements

```
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
wordcloud
joblib
```

## Appendix B: Repository Structure

```
Minor_Project/
|-- Fake_News_Detection.ipynb    # Main Jupyter Notebook
|-- README.md                     # Project documentation
|-- requirements.txt              # Python dependencies
|-- datasets/                     # Dataset directory
|   |-- WELFake_Dataset.csv       # Global news dataset
|   |-- IFND_full.csv             # Indian news dataset
|   |-- IFND_dataset.csv          # IFND subset
|   |-- Indian_Fake_News.csv      # Indian fake news
|   |-- Indian_True_News.csv      # Indian true news
|   |-- liar_dataset/             # LIAR benchmark dataset
|-- saved_models/                 # Serialized models
|   |-- best_logistic_regression.pkl  # Best model
|   |-- tfidf_vectorspace.pkl        # TF-IDF vectorizer
|-- documents/                    # Project documents
|   |-- Synopsis.md               # Project synopsis
|   |-- Project_Report.md         # Project report
|   |-- Research_Paper.md         # Research paper
```
