import tflearn
import tensorflow as tf

class Model:

    @staticmethod
    def gen_model(train_x, train_y):
        # Define the NN
        tf.reset_default_graph()

        # Describes the structure of the input that will be fed to the network
        ph = tf.placeholder(tf.float32, shape=[None, len(train_x[0])])

        # nn = tflearn.input_data(shape=[None, len(train_x[0])])
        nn = tflearn.input_data(placeholder=ph)
        # 2 hidden layer with 8 neurons each
        nn = tflearn.fully_connected(nn, 8)
        nn = tflearn.fully_connected(nn, 8)
        # TODO Why use softmax here?
        nn = tflearn.fully_connected(nn, len(train_y[0]), activation='softmax')
        # Regression at the end necessary because of TensorFlow (else tf collection 'trainops' is empty)
        nn = tflearn.regression(nn)

        # Define the model (Deep neural network)
        model = tflearn.DNN(nn, tensorboard_dir='logs', tensorboard_verbose=3)
        return model