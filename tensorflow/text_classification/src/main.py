from classifier.nn import Perceptron

from data import get_train_test_data
from features import *
from settings import *
from perceptron import *

import tensorflow as tf

def get_batch(data, target, word2index, input_size, iteration, batch_size):
    """ Defines the size of each batch to be processed. """
    batches = []
    results = []
    texts = data[iteration * batch_size : iteration * batch_size + batch_size]
    categories = target[iteration * batch_size : iteration * batch_size + batch_size]
    for text in texts:
        layer = np.zeros(input_size, dtype=float)
        for token in get_tokens(text):
            layer[word2index[token]] += 1
        batches.append(layer)
    for category in categories:
        y = np.zeros(len(set(categories)), dtype=float)
        y[category] = 1.0
        results.append(y)
    
    return np.array(batches), np.array(results)

if __name__ == '__main__':
    train_text, test_text = get_train_test_data(DATA_CATEGORIES)
    vocab = get_vocab(train_text.data + test_text.data)
    n_input = len(vocab)
    word2index = get_word_2_index(vocab)

    input_tensor = tf.placeholder(tf.float32, [None, n_input], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, len(DATA_CATEGORIES)], name="output")

    nn = Perceptron(n_input, len(DATA_CATEGORIES), N_HIDDEN, SIZE_HIDDEN)
    prediction = nn.predict(input_tensor)

    loss = get_entropy_loss(prediction, output_tensor)
    optimizer = get_optimizer(loss, LEARNING_RATE)

    # Initializing the variables
    init = tf.global_variables_initializer()

    training_epochs = 10
    # Launch the graph
    with tf.Session() as session:
        session.run(init) # inits the variables
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0
            BATCH_SIZE = len(train_text.data)
            total_batch = int(len(train_text.data) / BATCH_SIZE)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = get_batch(train_text.data, train_text.target, word2index, n_input, i, BATCH_SIZE)
                # Run optimization op (back propagation) and cost optimization
                c, _ = session.run(
                    [loss, optimizer],
                    feed_dict={
                        input_tensor: batch_x,
                        output_tensor: batch_y
                     }
                )
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % DISPLAY_STEP == 0:
                print("Epoch", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
    
        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        total_test_data = len(test_text.target)
        batch_x_test, batch_y_test = get_batch(test_text.data, test_text.target, word2index, n_input, 0, total_test_data)
        print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))