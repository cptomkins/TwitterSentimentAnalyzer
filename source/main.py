import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import logging

def validateModel(model, history, X_test_pad, y_test):
    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_test_pad, y_test)

    # Log the validation loss and accuracy
    logger.info(f"Validation Loss: {val_loss}")
    logger.info(f"Validation Accuracy: {val_accuracy}")

    # Plot the training history
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load the dataset with specified encoding
df = pd.read_csv('..\\dataset\\training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

# Extract the relevant columns
sentiments = df.iloc[:, 0]
tweets = df.iloc[:, 5]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tweets, sentiments, test_size=0.2, random_state=42)

# Log the number of samples
logger.info(f"Number of training samples: {len(X_train)}")
logger.info(f"Number of testing samples: {len(X_test)}")

# Define the maximum number of words to be used and the maximum length of each sequence
max_words = 10000
max_len = 100

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert the tweets to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Log the shape of the padded sequences
logger.info(f"Training data shape: {X_train_pad.shape}")
logger.info(f"Testing data shape: {X_test_pad.shape}")

# One-hot encode the sentiment labels
y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)

# Define the model with an LSTM layer for multi-class classification
model = Sequential([
    Embedding(input_dim=max_words, output_dim=16, input_length=max_len),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(5, activation='softmax')  # 5 units for 5 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Build the model by providing a dummy input
model.build(input_shape=(None, max_len))

# Log the model summary
model.summary(print_fn=logger.info)

# Train the model
history = model.fit(X_train_pad, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test_cat))


validateModel(model, history, X_test_pad, y_test)