from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def preprocess_text(text, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences

def predict_sentiment(model, text):

    max_words = 15000
    max_len = 50

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    processed_text = preprocess_text(text, tokenizer, max_len)
    prediction = model.predict(processed_text)
    sentiment = prediction.argmax(axis=-1)[0]
    return sentiment