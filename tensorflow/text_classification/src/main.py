from settings import *

from load.data import *
from data_transform.one_hot_encoder import OneHotEncoder
from classifier.nn import Perceptron
from model.model import Model

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    train_text, test_text = get_train_test_data(DATA_CATEGORIES)

    target_classes = np.unique(train_text.target)
    data = OneHotEncoder(train_text.data + test_text.data, target_classes)

    train_data = data.transform_input(train_text.data)
    train_target = data.transform_output(train_text.target)

    test_data = data.transform_input(test_text.data)
    test_target = data.transform_output(test_text.target)

    input_tensor = tf.placeholder(tf.float32, [None, data.get_vocab_length()], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, len(DATA_CATEGORIES)], name="output")

    nn = Perceptron(data.get_vocab_length(), len(data.target_classes), N_HIDDEN, SIZE_HIDDEN)

    parameters = {
        'learning_rate': LEARNING_RATE,
        'loss_threshold': 0.01
    }

    model = Model(nn, parameters, train_data, train_target, test_data, test_target)

    model.train(input_tensor, output_tensor)
    model.test(input_tensor, output_tensor)
