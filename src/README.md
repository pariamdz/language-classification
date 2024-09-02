# language classification in social networks Project
=============================================================================================

Language classification in social networks" refers to the process of identifying the languages used in social network content. Language Classification involves determining the language(s) in which the content is written and can be done using various techniques such as natural language processing (NLP), machine learning, or linguistic analysis. Understanding the diversity of languages used on social networks can help in providing multilingual support and content moderation.
This project focuses on language classification using machine learning techniques on social media text data. The goal is to preprocess text data, vectorize it using word embeddings, and classify the language of the text content.

## 'run.py file':

## Project Structure
- `preprocess`: Function for text preprocessing.
- `Word2VecVectorizer`: Implements a Word2VecVectorizer class for word embedding.
- `test_big_data`: Script for language classification on new data.
- `KNN_model.pkl`: Pre-trained K-Nearest Neighbors model for language classification.
- `insta_wchr.vec`: Word embeddings file used for text vectorization.
- `test.xlsx`: Sample test data in Excel format.

## Usage
1. Replace the `test.xlsx` with your data and then run the following command to test language classification on new data:
   ```
   python3 run.py
   ```
   This script reads data , preprocesses it, applies word embeddings using Word2Vec, and predicts the language using a pre-trained KNN model.

2. The predicted labels along with the original text content are saved in `label_test.xlsx`.

## Dependencies
- Python 3.8.18
- install requirements.txt
