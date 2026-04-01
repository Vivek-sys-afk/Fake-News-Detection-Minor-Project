# Fake News Detection System (Full Stack Data Science Pipeline)

### M.Sc. Data Science - Minor Project

Welcome to the **Fake News Detection** pipeline repository! This project represents a complete data science workflow built for detecting fake news. It includes robust Exploratory Data Analysis (EDA), advanced Natural Language Processing (NLP) visualizations, machine learning evaluations, and model persistence.

---

## Project Overview

This repository contains a comprehensive Jupyter Notebook that covers the end-to-end data science lifecycle. The goal of this project is to classify news articles as **Real** or **Fake** using Natural Language Processing and Machine Learning classification algorithms.

The workflow is broken into 7 main sections:
1. **Library Imports & Setup:** Installing and configuring required tools.
2. **Data Loading:** Safely loading local datasets (incorporating Global WELFake and Regional IFND datasets).
3. **Text Preprocessing:** Cleaning raw text data (Regex manipulation, Stopword Removal, and Lemmatization).
4. **Exploratory Data Analysis (EDA) & Visualization:** Graphing word counts, Class Distributions, Word Clouds, and N-gram frequencies.
5. **Feature Engineering:** Converting clean text into mathematical features using TF-IDF Vectorized representations.
6. **Model Training & Evaluation:** Training ML models (Logistic Regression, Naive Bayes, Random Forest) and visualizing their comparative ROC-AUC curves.
7. **Model Saving:** Serializing algorithms using `joblib` for future production.

---

## Repository Structure

- `Fake_News_Detection.ipynb`: The main Jupyter Notebook containing the entire pipeline.
- `datasets/`: Directory containing the datasets used in this project (`WELFake_Dataset.csv` and `IFND_full.csv`).
- `saved_models/`: Directory where the trained Machine Learning models and TF-IDF Vectorizer are serialized.
- `requirements.txt`: Python dependencies required to run the notebook.

---

## Getting Started

To run this project on your local machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Fake-News-Detection-Minor-Project.git
cd Fake-News-Detection-Minor-Project
```

### 2. Install Dependencies
Make sure you have Python installed. You can install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
Launch Jupyter Notebook to explore the code:
```bash
jupyter notebook Fake_News_Detection.ipynb
```

---

## Built With

- **Python** - The core programming language.
- **Pandas & NumPy** - Data manipulation and analysis.
- **NLTK (Natural Language Toolkit)** - Text preprocessing and NLP.
- **Scikit-Learn** - Machine Learning algorithms and feature extraction.
- **Matplotlib, Seaborn & WordCloud** - Data visualization.

---

## Key Visualizations
- Class Balance & Semantic Density (Word Counts)
- Fake vs. Real News Word Clouds
- Top 20 N-Grams (Unigrams and Bigrams)
- Model Confusion Matrices
- ROC-AUC Curve Comparisons

---

## License

This project is part of an M.Sc. Data Science academic curriculum. Feel free to use and modify it for educational purposes.
