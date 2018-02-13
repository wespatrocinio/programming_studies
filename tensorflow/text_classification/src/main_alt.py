from data import get_train_test_data
from features import *
from settings import *
from perceptron import *

from classifier.nn import Perceptron

import tensorflow as tf



if __name__ == '__main__':
    train_text, test_text = get_train_test_data(DATA_CATEGORIES)
    vocab = get_vocab(train_text.data + test_text.data)
    n_input = len(vocab)
    word2index = get_word_2_index(vocab)

    input_tensor = tf.placeholder(tf.float32, [None, n_input], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, len(DATA_CATEGORIES)], name="output")

    nn = Perceptron(n_input, len(DATA_CATEGORIES), N_HIDDEN, SIZE_HIDDEN)
    prediction = nn.predict(input_tensor)

    
    
        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        total_test_data = len(test_text.target)
        batch_x_test, batch_y_test = get_batch(test_text.data, test_text.target, word2index, n_input, 0, total_test_data)
        print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))