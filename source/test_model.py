from tensorflow.keras.models import load_model
import testing_utils


if __name__=='__main__':
    # Load the saved model
    model_path = input("\nPlease provide the path to the model:\n")

    model = load_model(model_path)

    while True:

        text = input("\nType a phrase to analyze sentiment:\n")
        if text == 'exit':
            break

        answer = testing_utils.predict_sentiment(model, text)

        print(answer)