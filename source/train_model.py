# ----- Third Party Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dropout, Dense
from tensorflow.keras.utils import to_categorical

# ----- Python Imports
import logging
import argparse
import os

# ----- Package Imports
from training_utils import *

def run(log_level=logging.INFO):
    logger = getLogger(level=log_level)

    logger.info("##### BEGIN #####")

    logger.info("Loading dataset...")
    raw_data_path = '../dataset/training.1600000.processed.noemoticon.csv'
    
    df = pd.read_csv(raw_data_path, encoding='ISO-8859-1')
    logger.debug("Dataset loaded.")

    logger.info("Extracting columns...")
    sentiments = df.iloc[:, 0]
    tweets = df.iloc[:, 5]
    logger.debug("Columns extracted.")

    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(tweets, sentiments, test_size=0.2, random_state=42)
    logger.debug("Data split into training and testing sets.")

    logger.info(f"Number of training samples: {len(X_train)}")
    logger.info(f"Number of testing samples: {len(X_test)}")

    max_words = 15000
    max_len = 50

    logger.info("Initializing tokenizer...")
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    logger.debug("Tokenizer initialized and fitted.")

    logger.info("Converting texts to sequences...")
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    logger.debug("Texts converted to sequences.")

    logger.info("Padding sequences...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    logger.debug("Sequences padded.")

    logger.info(f"Training data shape: {X_train_pad.shape}")
    logger.info(f"Testing data shape: {X_test_pad.shape}")

    logger.info("One-hot encoding labels...")
    y_train_cat = to_categorical(y_train, num_classes=5)
    y_test_cat = to_categorical(y_test, num_classes=5)
    logger.debug("Labels one-hot encoded.")

    logger.info("Defining the model...")
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
        GRU(128, return_sequences=True),
        Dropout(0.5),
        GRU(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')
    ])
    logger.debug("Model defined.")

    logger.info("Compiling the model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    logger.debug("Model compiled.")

    model.summary(print_fn=logger.info)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    logger.info("Training the model...")
    history = model.fit(X_train_pad, y_train_cat, epochs=20, batch_size=64, validation_data=(X_test_pad, y_test_cat), callbacks=[early_stopping])
    logger.debug("Model trained.")

    logger.info("Saving the model...")
    model.save('sentiment_model.h5')
    logger.debug("Model Saved.")

    evaluateModel(model, history, X_test_pad, y_test_cat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ai Model to predict tweet sentiment rating as 0 Negative, 2 Neutral, or 4 Positive.")

    parser.add_argument(
        '--debug',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='Set the debug level (0 = Warnings Only, 1 = Warnings and Info, 2 = Debugging)'
    )

    parser.add_argument(
        '-dir', '--directory',
        type=str,
        default="../model/",
        help='Path to the model directory (default: "../model/")',
        dest='directory'
    )
    parser.add_argument(
        '-fn', '--file_name',
        type=str,
        default="sentiment_model",
        help='Name of the model file (default: "sentiment_model")',
        dest='file_name'
    )

    args = parser.parse_args()

    if os.name == 'nt':  # 'nt' indicates Windows
        args.directory = args.directory.replace('/', '\\')

    if args.debug == 0:
        level = logging.WARNING
    elif args.debug == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    run(file_name=args.file_name, directory=args.directory,log_level=level)
