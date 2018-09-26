# Code is based on the Tutorial: https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

import sys
# Tensorflow imports
import numpy as np
import tflearn
import tensorflow as tf
import random

from filehelper import FileHelper
from preprocessor import Preprocessor
from training import Training
from model import Model
from response_generator import ResponseGenerator

class Bot:

    def train(self):
        intents = FileHelper.intents_load()
        preprocessor = Preprocessor()
        words, classes, documents = preprocessor.extract_info_from_intents(intents)
        print('Extracted from intents %d documents, %d classes and %d unique words' % (
            len(documents), len(classes), len(words)))

        # Tensorflow will take train_x to build the model and train_y to minimize the cost/loss function
        train_x, train_y = preprocessor.convert_intents_to_tensors(words, classes, documents)

        FileHelper.training_data_save(words, classes, train_x, train_y)

        training = Training()
        training.train(train_x, train_y)

    def start(self):
        # Load the data and model
        self.intents = FileHelper.intents_load()
        self.words, self.classes, self.train_x, self.train_y = FileHelper.training_data_load()
        self.model = Model.gen_model(self.train_x, self.train_y)
        self.model.load('model.tflearn')

        self.respgen = ResponseGenerator(self.intents, self.words, self.classes, self.model)

    def ask(self, input_sentence, userId):
        return self.respgen.gen_response(input_sentence, userId)
