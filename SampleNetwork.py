# This is a simple script showing how to set up and use the network.
# Set the 'GPU' flag to run the training on the gpu.
# Note: The device can also be set in the .theanorc initialization file.
#       For more information see the theano's website.


# Training, validation and test data are m x n matrices with m samples of n elements.
# Training, validation and test targets are 1 x m vectors where each element shows the class of the respective training/
# validation/test sample.
# More details about the format of the data can be found in the MakeData.py file.
# Use the MakeData.shared() function to set the data types.


import theano
import LayeredNetwork
import FullyConnectedLayer
import SoftmaxLayer
import ConvolutionalLayer
import loadDataSets
import activations

# Setting the device
GPU = True
if GPU:
    print "Trying to run under a GPU."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU."

print "\nRunning on " + theano.config.device + "...\n"

# loading a sample dataset

trainingData, validationData, testData = loadDataSets.loadMnist()


# setting the mini_batch_size

sizeOfMiniBatch = 300

CNN = LayeredNetwork.Network([ConvolutionalLayer.ConvPoolLayer(         image_shape     = (sizeOfMiniBatch, 1, 28, 28),
                                                                        filter_shape    = (20, 1, 5, 5),
                                                                        conv_type='valid',
                                                                        poolsize        = (2, 2),
                                                                        activation_fn   = activations.elu),
                              ConvolutionalLayer.ConvPoolLayer(         image_shape     = (sizeOfMiniBatch, 20, 12, 12),
                                                                        filter_shape    = (40, 20, 3, 3),
                                                                        conv_type='valid',
                                                                        poolsize        = (2, 2),
                                                                        activation_fn   = activations.elu),
                              FullyConnectedLayer.FullyConnectedLayer(  n_in            = 40*5*5,
                                                                        n_out           = 1000,
                                                                        activation_fn   = activations.absolute),
                              SoftmaxLayer.SoftmaxLayer(                n_in            = 1000,
                                                                        n_out           = 10)]
                              , sizeOfMiniBatch)

CNN.SGD(training_data=trainingData,validation_data=validationData, test_data=testData, mini_batch_size=sizeOfMiniBatch, epochs=20, eta=0.02, lmbd=0.0)

# LayeredNetwork.saveNet(CNN, 'sampleCNN')
#
# P = LayeredNetwork.loadNet("sampleCNN.p")
