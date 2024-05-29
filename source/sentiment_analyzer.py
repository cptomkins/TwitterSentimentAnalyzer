import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import logging
import os
from training_utils import getLogger


class SentimentAnalyzer:
    def __init__(self, max_words=15000, max_len=50, debug=False):
        self.max_words = max_words
        self.max_len = max_len
        self.logger = getLogger(__name__, level=logging.DEBUG if debug else logging.INFO)
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None
        self.debug = debug

    def load_data(self, data_path='../dataset/training.1600000.processed.noemoticon.csv'):
        df = pd.read_csv(data_path, encoding='ISO-8859-1')
        sentiments = df.iloc[:, 0]
        tweets = df.iloc[:, 5]
        return train_test_split(tweets, sentiments, test_size=0.2, random_state=42)
    
    def preprocess_data(self, X_train, X_test):
        self.tokenizer.fit_on_texts(X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        return pad_sequences(X_train_seq, maxlen=self.max_len), pad_sequences(X_test_seq, maxlen=self.max_len)

    def build_model(self):
        self.model = Sequential([
            Embedding(input_dim=self.max_words, output_dim=128, input_length=self.max_len),
            GRU(128, return_sequences=True),
            Dropout(0.5),
            GRU(64),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(5, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
        if self.model is None:
            self.build_model()

        if self.debug:
            self.logger.warning('Analyzer set to debug, Skipping model train.')
            return None
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        y_train_cat = to_categorical(y_train, num_classes=5)
        y_test_cat = to_categorical(y_test, num_classes=5)
        return self.model.fit(X_train, y_train_cat, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_cat), callbacks=[early_stopping])

    def evaluate(self, X_test, y_test, history):
        val_loss, val_accuracy = self.model.evaluate(X_test, y_test)
        self.logger.info(f"Validation Loss: {val_loss}")
        self.logger.info(f"Validation Accuracy: {val_accuracy}")

        history_dict = history.history
        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 9))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def save(self, file_name, directory='../model/'):
        model_path = os.path.join(directory, file_name)
        os.makedirs(directory, exist_ok=True)
        self.model.save(model_path) 

    def predict_sentiment(self, text):
        """
        Predicts the sentiment of a given text string.

        Args:
            text: The text string to analyze.

        Returns:
            The predicted sentiment (0, 2, or 4).
        """

        if self.model is None:
            self.logger.error("Model not trained yet. Call train() first.")
            return None

        sequences = self.tokenizer.texts_to_sequences([text])  # No need for a new Tokenizer
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        prediction = self.model.predict(padded_sequences)
        sentiment = prediction.argmax(axis=-1)[0] 
        return sentiment  


if __name__=='__main__':
    sa = SentimentAnalyzer()
    sa.logger.info('\nHELLO')
