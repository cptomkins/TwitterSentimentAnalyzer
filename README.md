# Tweet Sentiment Analysis

This project involves the creation of a machine learning model to classify the sentiment of tweets. The script `main.py` preprocesses the tweet data, trains a neural network model, and evaluates its performance.

## Overview

The `main.py` script performs the following tasks:

1. **Loading Dataset**: Loads the Sentiment140 dataset from a CSV file.
2. **Data Cleaning**: Cleans the tweets by removing punctuation, numbers, URLs, and stopwords, and converting the text to lowercase.
3. **Saving Cleaned Data**: Saves the cleaned tweets to a file to avoid re-cleaning in subsequent runs.
4. **Data Splitting**: Splits the dataset into training and testing sets.
5. **Text Tokenization**: Converts the text data to sequences of integers.
6. **Padding Sequences**: Pads the sequences to ensure uniform input length for the model.
7. **Model Definition**: Defines a neural network model with embedding, LSTM, dropout, and dense layers.
8. **Model Training**: Trains the model using the training data.
9. **Model Evaluation**: Evaluates the model's performance on the test data and plots training history.

## Dataset

The dataset used in this project is Sentiment140, which can be found at the following link:

[Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Requirements

The following libraries are required to run the `main.py` script:

- pandas
- numpy
- sklearn
- tensorflow
- nltk
- matplotlib
- logging
- re
- os

You can install the required packages using the following command:

```sh
pip install pandas numpy scikit-learn tensorflow nltk matplotlib
```

Additionally, you need to download the NLTK stopwords by running the following Python code once:
```python
import nltk
nltk.download('stopwords')
```

# Running the script
To run the script, execute the following command in your terminal:
```bash
python main.py
```

This `README.md` provides an overview of the `main.py` file, links to the dataset, lists the required packages, and includes instructions for running the script.
