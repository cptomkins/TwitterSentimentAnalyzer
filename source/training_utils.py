# ----- 3rd Party Imports
import matplotlib.pyplot as plt

# ----- Python Imports
import logging
import os

def getLogger(level=logging.INFO):
    """ Setup Logging
    """
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger

def evaluateModel(model, history, X_test_pad, y_test, logger=getLogger()):
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

def saveModel(file_name, directory, model):
    # Define the directory and file path
    model_path = os.path.join(directory, f"{file_name}.h5")

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the model
    model.save(model_path)