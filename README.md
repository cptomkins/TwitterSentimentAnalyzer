# Twitter Sentiment Analyzer

This package leverages TensorFlow to train a model capable of analyzing the sentiment of text. It includes a script for training the model `train_model.py` and a separate script for testing `test_model.py`. there is an example model `sentiment_model.h5` included.

## Overview

### train_model.py

This script is responsible for loading the dataset, preprocessing the data, building and training the model, and saving the trained model.

### test_model.py

This script loads the saved model and allows you to input text snippets to predict their sentiment.

### training_utils.py

Contains utility functions for logging setup.

### sentiment_analyzer.py

Defines the `SentimentAnalyzer` class, which includes methods for loading data, preprocessing, building, training, evaluating, saving the model, and predicting sentiment.

## Dataset

The dataset used in this project is Sentiment140, which can be found at the following link:

[Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Requirements

The following libraries are required to run the `train_model.py` script:

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- logging
- os

You can install the required packages using the following command:

```sh
pip install pandas numpy scikit-learn tensorflow matplotlib
```

## Running the Script

### Training the Model

To train the model, execute the following command in your terminal:

```bash
python train_model.py
```

### Testing the Model

To test the trained model on any snippet of text, run the following command:

```bash
python test_model.py
```

---

This `README.md` provides an overview of the project, links to the dataset, lists the required packages, and includes instructions for running the training and testing scripts.
