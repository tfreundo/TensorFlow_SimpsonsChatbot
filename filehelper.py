import json
import pickle

class FileHelper:

    @staticmethod
    def intents_load():
        """Loads the data from a JSON file
        """
        data = ""
        with open('data/intents.json') as f:
            data = json.load(f)
        return data

    @staticmethod
    def training_data_save(words, classes, train_x, train_y):
        """Saves the training data to a file
        """
        training_file = open("data/training", "wb")
        pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, training_file)

    @staticmethod
    def training_data_load():
        """Loads the training data from a file
        """
        training_file = open("data/training", "rb")
        data = pickle.load(training_file)
        return data['words'], data['classes'], data['train_x'], data['train_y'] 