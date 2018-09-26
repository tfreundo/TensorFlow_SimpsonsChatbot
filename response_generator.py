import numpy as np
import random
# Natural Language Processing (NLP)
import nltk
# Stemmer that extracts the stem/core of a word (e.g. planning --> stem is plan)
from nltk.stem.lancaster import LancasterStemmer

class ResponseGenerator:

    # The minimal probability that is necessary to use a prediction as an answer
    ERR_THRESHOLD = 0.33

    def __init__(self, intents, words, classes, model):
        self.stemmer = LancasterStemmer()
        self.intents = intents
        self.classes = classes
        self.words = words
        self.model = model

    def tokenize_stem_sentence(self, input_sentence):
        """Tokenizes and stems the words of a input sentence and returns the words
        """
        words = nltk.word_tokenize(input_sentence)
        words = [self.stemmer.stem(word.lower()) for word in words]

        return words

    def gen_bag_of_words(self, sentence, debug=False):
        """Generates the bag of words based on the words learned from intents and the input
        sentence. Will return a numpy array that has a 1 for every word found in the input sentence
        and the intents (0 if not).
        """
        input_words = self.tokenize_stem_sentence(sentence)
        
        bow = [0]*len(self.words)
        
        for x in input_words:
            for i,w in enumerate(self.words):
                # Check if the word stem is in the knowns words (from intents)
                if w == x:
                    # Found a word, mark with 1
                    bow[i] = 1
                    if debug:
                        print('Found a match for word: %s' %w )
        
        np_arr = np.array(bow)

        return np_arr

    def classify_input(self, input_sentence, debug=False):
        """Classifies a given input sentence into classes and their probabilities
        """
        # calculate probabilities
        bow = self.gen_bag_of_words(input_sentence)

        predictions = self.model.predict([bow])[0]

        # Check if predictions are somehow certain (above the defined threshold)
        predictions = [[i,p] for i,p in enumerate(predictions) if p > self.ERR_THRESHOLD]
        # Sort by probability descending
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Form the result
        result = []
        for p in predictions:
            result.append((self.classes[p[0]], p[1]))
        
        if debug:
            print('Predictions for each class:', result)

        return result

    # TODO Handle multiple users (necessary for handling contextual conversations)
    def gen_response(self, input_sentence, userId):
        response_result = self.classify_input(input_sentence, debug=True)

        response = ""

        if response_result:
            while response_result:
                for intent in self.intents['intents']:
                    # Find a matching tag
                    if intent['tag'] == response_result[0][0]:
                        # Choose randomly one of the defined responses
                        return random.choice(intent['responses'])
                # Remove this item from results
                response_result.pop(0)
        else:
            # TODO Generate a set of sentences (from intents) like 'I didnt understand you. Could you blabla' 
            response = 'Sorry, I did not understand you. Could you reformulate your question?'
            return response