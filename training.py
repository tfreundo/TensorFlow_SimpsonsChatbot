import tflearn
import tensorflow as tf
from model import Model

class Training:

    def train(self, train_x, train_y):
        model = Model.gen_model(train_x, train_y)
        model.fit(train_x, train_y, n_epoch=1000, show_metric=True)
        model.save('model.tflearn')