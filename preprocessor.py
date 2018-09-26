# Natural Language Processing (NLP)
import nltk
# Stemmer that extracts the stem/core of a word (e.g. planning --> stem is plan)
from nltk.stem.lancaster import LancasterStemmer
import random
import numpy as np

class Preprocessor:

    def __init__(self):
        self.stemmer = LancasterStemmer()
        self.ignore_words = ['?']  # List of words to ignore

    def extract_info_from_intents(self, intents):
        """Extracts/splits the information from the intents
        """
        words = []
        classes = []
        docs = []

        # Extract/split the information from the intents
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize the words
                word = nltk.word_tokenize(pattern)
                words.extend(word)
                docs.append((word, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        # Stem the words and convert them to lower case
        words = [self.stemmer.stem(word.lower())
                 for word in words if word not in self.ignore_words]
        # Remove potential duplicates and sort it
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))

        return words, classes, docs

    def convert_intents_to_tensors(self, words, classes, documents):
        """Converts the information extracted from the intents (words, sentences, ...) into numbers to use for learning
        """
        train_all = []
        output = []
        output_empty = [0] * len(classes)

        # Convert to a bag of words (bow)
        for doc in documents:
            bow = []
            pattern_words = doc[0]
            # Stem the words
            pattern_words = [self.stemmer.stem(word.lower()) for word in pattern_words]
            for w in words:
                # 0 for each tag and 1 for current tag
                bow.append(1) if w in pattern_words else bow.append(0)

            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            train_all.append([bow, output_row])

        # shuffle the data
        random.shuffle(train_all)
        # Convert to a Numpy array
        train_all = np.array(train_all)

        # Split into training and test data set
        train_x = list(train_all[:,0])
        train_y = list(train_all[:,1])

        return train_x, train_y