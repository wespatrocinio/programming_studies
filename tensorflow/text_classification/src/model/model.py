import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, classifier, parameters, train_data, train_target, test_data, test_target):
        self.classifier = classifier
        self.parameters = parameters
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target

    def train(self, input_tensor, output_tensor):
        loss = self.get_entropy_loss(self.classifier.predict(input_tensor), output_tensor)
        optimizer = self.get_optimizer(loss, self.parameters.get('learning_rate'))

        # Initializing the variables
        init = tf.global_variables_initializer()

        training_epochs = 10
        # Launch the graph
        with tf.Session() as session:
            session.run(init) # inits the variables
            # Training cycle
            epoch = 0
            cost = 1 # Any value bigger than the threshold
            while cost > 0.01 or epoch <= training_epochs: # TODO: parameterize the cost threshold
                cost, _ = session.run(
                    [loss, optimizer],
                    feed_dict={
                        input_tensor: np.array(self.train_data),
                        output_tensor: np.array(self.train_target)
                    }
                )
                # Display logs per epoch step
                print("Epoch", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(cost))
            print("Optimization Finished!")
        return True
    
    def test(self):
        pass
    
    # def get_batch(data, target, word2index, input_size, iteration, batch_size):
    #     """ Defines the size of each batch to be processed. """
    #     batches = []
    #     results = []
    #     texts = data[iteration * batch_size : iteration * batch_size + batch_size]
    #     categories = target[iteration * batch_size : iteration * batch_size + batch_size]
    #     for text in texts:
    #         layer = np.zeros(input_size, dtype=float)
    #         for token in get_tokens(text):
    #             layer[word2index[token]] += 1
    #         batches.append(layer)
    #     for category in categories:
    #         y = np.zeros(len(set(categories)), dtype=float)
    #         y[category] = 1.0
    #         results.append(y)
        
    #     return np.array(batches), np.array(results)

    def get_entropy_loss(self, prediction, output_tensor):
        """ Calculate the mean loss based in the difference between the prediction and the ground truth. """
        entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
        return tf.reduce_mean(entropy_loss)

    def get_optimizer(self, loss, learning_rate):
        """ Update all the variables bases on the Adaptive Moment Estimation (Adam) method. """
        return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)