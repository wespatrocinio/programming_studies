from settings import *

import tensorflow as tf

def get_weights(n_input, n_hidden_1, n_hidden_2, n_classes):
    """ Generate all weights based on the NN settings. """
    return {
        'h0': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h1': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }

def get_biases(n_hidden_1, n_hidden_2, n_classes):
    """ Generate all biases based on the NN settings. """
    return {
        'h0': tf.Variable(tf.random_normal([n_hidden_1])),
        'h1': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

def multilayer_perceptron(input_tensor, weights, biases):
    """
    Creates the multi-layer NN by applying weights and biases for each layer.

    input_tensor    Input data (phrase)
    weights         Weights for each layer
    biases          Biases for each layer
    """
    layer_1_multiplication = tf.matmul(input_tensor, weights['h0'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['h0'])
    layer_1_activation = tf.nn.relu(layer_1_addition)

    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h1'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['h1'])
    layer_2_activation = tf.nn.relu(layer_2_addition)

    # Output layer with linear activiation
    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition

def get_entropy_loss(prediction, output_tensor):
    """ Calculate the mean loss based in the difference between the prediction and the ground truth. """
    entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
    return tf.reduce_mean(entropy_loss)

def get_optimizer(loss, learning_rate):
    """ Update all the variables bases on the Adaptive Moment Estimation (Adam) method. """
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)