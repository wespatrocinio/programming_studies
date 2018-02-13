from settings import *

from classifier.nn import Perceptron
from data_transform.one_hot_encoder import OneHotEncoder
from load.data import *

import tensorflow as tf
import numpy as np



if __name__ == '__main__':
    train_text, test_text = get_train_test_data(DATA_CATEGORIES)

    target_classes = np.unique(train_text.target)
    data = OneHotEncoder(train_text.data + test_text.data, target_classes)

    train_data = data.transform_input(train_text.data)
    train_target = data.transform_output(train_text.target)

    input_tensor = tf.placeholder(tf.float32, [None, data.get_vocab_length()], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, len(DATA_CATEGORIES)], name="output")

    nn = Perceptron(data.get_vocab_length(), len(data.target_classes), N_HIDDEN, SIZE_HIDDEN)
    prediction = nn.predict(input_tensor)    
    
    # # Test model
    # correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # total_test_data = len(test_text.target)
    # batch_x_test, batch_y_test = get_batch(test_text.data, test_text.target, word2index, n_input, 0, total_test_data)
    # print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))