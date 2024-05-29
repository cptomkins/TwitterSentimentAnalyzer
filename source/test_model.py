from tensorflow.keras.models import load_model
from sentiment_analyzer import SentimentAnalyzer


if __name__=='__main__':
    # Load the saved model
    model_path = '../model/sentiment_model.h5'

    analyzer = SentimentAnalyzer()
    X_train, X_test, y_train, y_test = analyzer.load_data()
    analyzer.initTokenizer(X_train)
    analyzer.model = load_model(model_path)

    while True:

        text = input("\nType a phrase to analyze sentiment:\n")
        if text == 'exit':
            break

        answer = analyzer.predict_sentiment(text)

        print(answer)