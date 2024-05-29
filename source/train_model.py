# ----- Python Imports
import logging
import argparse
import os

# ----- Package Imports
from training_utils import *
from sentiment_analyzer import SentimentAnalyzer

def run(file_name='sentiment_model.h5', directory='../model/', log_level=logging.INFO):
    logger = getLogger(name=__name__, level=log_level)

    logger.info("Initializing Sentiment Analyzer")
    analyzer = SentimentAnalyzer(max_words=15000, max_len=50, debug=False)

    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test = analyzer.load_data()
    logger.debug("Data loaded and split into training and testing sets.")

    logger.info(f"Number of training samples: {len(X_train)}")
    logger.info(f"Number of testing samples: {len(X_test)}")

    logger.info("Preprocess Data...")
    X_train_pad, X_test_pad = analyzer.preprocess_data(X_train, X_test)
    logger.debug("Data Preprocessed.")

    logger.info(f"Training data shape: {X_train_pad.shape}")
    logger.info(f"Testing data shape: {X_test_pad.shape}")

    logger.info("Training the model...")
    history = analyzer.train(X_train_pad, y_train, X_test_pad, y_test, epochs=20, batch_size=64)
    logger.debug("Model trained.")

    if not history:
        logger.warning('No history, Skipping model save and evaluation.')
        return None

    logger.info("Saving the model...")
    analyzer.save(file_name=file_name, directory=directory)
    logger.debug("Model Saved.")

    analyzer.evaluate(X_test, y_test, history)

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
