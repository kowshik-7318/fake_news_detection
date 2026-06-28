# 📰 Fake News Detection Using NLP

## 📌 Project Overview

Fake News Detection Using NLP is a Machine Learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques. The project preprocesses textual data, converts it into numerical features using TF-IDF Vectorization, trains multiple machine learning models, compares their performance, and selects the best model for prediction.

---

## 🚀 Features

* Load and explore news dataset
* Text preprocessing using NLP
* TF-IDF Vectorization
* Train multiple Machine Learning models
* Compare model performance
* Automatic best model selection
* Save trained model and vectorizer
* Predict custom news articles
* Colorful menu-driven terminal interface
* Save confusion matrix visualization

---

## 🛠 Technologies Used

* Python 3.13
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Joblib
* Colorama

---

## 🤖 Machine Learning Models

* Logistic Regression
* Multinomial Naive Bayes
* Random Forest
* Linear Support Vector Machine (Linear SVM)

---

## 🧠 NLP Techniques Used

* Lowercase Conversion
* Punctuation Removal
* Stop Word Removal
* Tokenization
* Lemmatization
* TF-IDF Vectorization

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── fake_news_detection.py
├── news.csv
├── requirements.txt
├── README.md
├── best_model.pkl
├── tfidf_vectorizer.pkl
├── confusion_matrix.png
├── data_analysis.png
└── fake_news_detection.log
```

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python fake_news_detection.py
```

---

## 📋 Project Workflow

1. Load and explore the dataset.
2. Preprocess the news text using NLP.
3. Convert text into numerical features using TF-IDF.
4. Train and evaluate multiple machine learning models.
5. Select and save the best-performing model.
6. Load the saved model for future use.
7. Predict whether a news article is Fake or Real.

---

## 📊 Output

The application displays:

* Dataset statistics
* Text preprocessing results
* Model comparison table
* Accuracy, Precision, Recall and F1 Score
* Confusion Matrix
* Prediction result
* Fake and Real probabilities
* Confidence level

---

## 📁 Generated Files

* `best_model.pkl` – Trained Machine Learning model
* `tfidf_vectorizer.pkl` – Saved TF-IDF vectorizer
* `confusion_matrix.png` – Model evaluation visualization
* `data_analysis.png` – Dataset analysis visualization
* `fake_news_detection.log` – Application log file

---

## 🎯 Future Enhancements

* Use a larger real-world news dataset
* Deep Learning models (LSTM/BERT)
* Real-time news prediction
* Web application using Flask or Streamlit
* REST API integration
* News URL prediction

---

## 👨‍💻 Author

**Kowsik**

Machine Learning & NLP Major Project

---

## ⭐ Repository

If you found this project helpful, consider giving it a ⭐ on GitHub.
