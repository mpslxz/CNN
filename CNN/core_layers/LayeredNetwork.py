import numpy as np
import theano
import theano.tensor as T
import time
import pickle

#   Main CNN class:
#   Class constructor:
#   Network([ layer_1, layer_2, ... , layer_n], mini_batch_size)
#       layer_i can be 1) convolutional-pooling, 2) fully-connected, 3) softmax layer

class Network(object):

    def __init__(self, layers, mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params if param is not None]
        self.x = T.matrix("x")
        self.y = T.ivector("y")

        print "Building the model.\n"
        # Setting the initial layer
        init_layer = self.layers[0]

        # input to the initial layer is the image
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)

        # Connecting the output of each layer to the input of the next layer
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                    prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    # Defining the Stochastic Gradient Descent
    # arguments:
    #   training_data   : training dataset
    #   validation_data : validation dataset
    #   test_data       : test dataset
    #   epochs          : number of training epochs
    #   mini_batch_size : size of the training batch
    #   eta             : learning rate
    #   lmbd            : regularization parameter

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbd=0.0):

        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # Calculating the number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # Defining the l2 norm regularization
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers if layer.trainable == True])

        # Defining the cost function
        cost = self.layers[-1].cost(self)+\
               0.5*lmbd*l2_norm_squared/num_training_batches

        # Defining the symbolic gradient
        grads = T.grad(cost, self.params)

        # Updating the parameters with the symbolic gradient
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # defining functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.

        # mini-batch index
        i = T.lscalar()
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # Calculating the SGD for all of the epochs

        print "Training the model.\n"
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}\n".format(iteration))
                train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy > best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))

        # Generating a log file to keep track of the accuracies
        print "Finished training. Writing log file.\n"
        with open("Training Log.txt","a") as logFile:
            logFile.write(time.strftime("%d/%m/%Y") + ": Finished training network at " + time.strftime("%H:%M:%S") + "\n")
            logFile.write("Best validation accuracy of {0:.2%} obtained at iteration {1}\n".format(
            best_validation_accuracy, best_iteration))
            logFile.write("Corresponding test accuracy of {0:.2%}\n".format(test_accuracy))

    # Defining the predict function for later predictions
    def predict(self, testData):
        counter = T.lscalar()
        predFcn = theano.function([counter], self.layers[-1].y_out, givens={self.x: testData[counter*self.mini_batch_size:(counter+1)*self.mini_batch_size]})
        nb_samples = testData.shape.eval()[0]
        result = np.zeros((nb_samples))

        for i in range(0,(nb_samples/self.mini_batch_size)):
            result[i*self.mini_batch_size:(i+1)*self.mini_batch_size] = predFcn(i)
        return result

    def get_layer_params(self, layer):
        return [self.layers[layer].w.get_value(), self.layers[layer].b.get_value()]


# Defining the save function to write the trained network on the HDD
def saveNet(network, path):
    print "Saving the model.\n"
    try:
        netFile = file(path+".p", "w")
        pickle.dump(network, netFile)
    except pickle.PicklingError:
        print "Error in saving the network."

# Defining the load function to load a pretrained network
def loadNet(path):
    print "Loading the model.\n"
    netFile = file(path,'rb')
    net = []
    try:
        net = pickle.load(netFile)
    except pickle.UnpicklingError:
        print "Error in loading the network."
    return net

def size(data):
    return data[0].get_value(borrow=True).shape[0]