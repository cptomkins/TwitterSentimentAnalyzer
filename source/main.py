import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dropout, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def evaluateModel(model, history, X_test_pad, y_test):
    val_loss, val_accuracy = model.evaluate(X_test_pad, y_test)
    logger.info(f"Validation Loss: {val_loss}")
    logger.info(f"Validation Accuracy: {val_accuracy}")

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

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

    max_words = 15000  # Increase the vocabulary size
    max_len = 50  # Adjust the sequence length

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
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),  # Increased embedding dimension
        GRU(128, return_sequences=True),  # Changed LSTM to GRU
        Dropout(0.5),  # Increased dropout rate
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

    evaluateModel(model, history, X_test_pad, y_test_cat)
