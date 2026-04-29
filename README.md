# Fake News Detection System (Full Stack Data Science Pipeline)

### M.Sc. Data Science - Minor Project

Welcome to the **Fake News Detection** pipeline repository! This project represents a complete data science workflow built for detecting fake news. It includes robust Exploratory Data Analysis (EDA), advanced Natural Language Processing (NLP) visualizations, machine learning evaluations, and model persistence.

---

## Project Overview

This repository contains a comprehensive dual-model data science pipeline. The goal of this project is to classify news articles as **Real** or **Fake** using both traditional Machine Learning and state-of-the-art Deep Learning (Transformers).

### Models Included:
1. **Transformer Model (`Fake_News_Detection.ipynb`)**: Leverages the **DistilBERT** architecture for semantic understanding and high-accuracy detection.
2. **Traditional ML Model (`Fake_News_Detection_Traditional.ipynb`)**: A robust legacy pipeline using **TF-IDF** features and an ensemble of **Logistic Regression, Passive Aggressive, and Linear SVC**.

---

## Repository Structure

- `Fake_News_Detection.ipynb`: Main notebook featuring the DistilBERT Transformer model.
- `Fake_News_Detection_Traditional.ipynb`: Legacy notebook featuring the TF-IDF and Stacking Ensemble models.
- `documents/`: Contains the project `Synopsis.md` and `Project_Report.md`.
- `datasets/`: Directory for `WELFake_Dataset.csv` and `IFND_full.csv`.
- `saved_models/`: Serialized models and vectorizers.
- `requirements.txt`: Python dependencies for both models.

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
- **Scikit-Learn** - Machine Learning algorithms, Feature Extraction (`FeatureUnion`, `TfidfVectorizer`), and Ensemble architectures (`StackingClassifier`, Probability Calibration).
- **Matplotlib, Seaborn & WordCloud** - Data visualization.

---

## Key Visualizations

- Class Balance & Semantic Density (Word Counts)
- Fake vs. Real News Word Clouds
- Top 20 N-Grams (Unigrams and Bigrams)
- Model Confusion Matrices
- ROC-AUC Curve Comparisons
- Individual Models vs. Ensemble Stacking Accuracy Profiles

---

## License

This project is part of an M.Sc. Data Science academic curriculum. Feel free to use and modify it for educational purposes.
