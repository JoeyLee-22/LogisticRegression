#[row][column]
import numpy as np
import time
import os
from array import array

class NeuralNetwork():

    width = 35
    numNewSituation = 10
    learning_rate = 0.1
    epochs = 10000
    sizeOfEpoch = 5

    def __init__(self, size):
        self.dimensions = size

        self.output = np.empty(size[1])

        self.weights = np.random.rand(size[0])
        self.bias = np.random.rand(1)

        self.weights_summation = np.zeros(size[0])      
        self.bias_summation = np.zeros(size[1])

        self.error = np.empty(size[1])

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
   
    def forwardProp(self, inputs):
        self.output = self.sigmoid(np.dot(inputs, self.weights)+self.bias)

    def backProp(self, inputs):
        for i in range (self.dimensions[0]):
            if i == 0:
                self.bias_summation += self.error
            self.weights_summation[i] += self.error*inputs[i]

    def change(self):
        self.weights -= self.weights_summation*self.learning_rate
        self.bias -= self.bias_summation*self.learning_rate

    def train(self, training_inputs, training_outputs):
        size = str(self.sizeOfEpoch)
        largest_error = 0

        for m in range(self.sizeOfEpoch):
            self.forwardProp(training_inputs[m])
            self.error = np.subtract(self.output, training_outputs[m])
            self.backProp(training_inputs[m])

            print("Error: ", round(np.absolute(self.error[0]), 6))
        self.change()
        self.weights_summation = np.zeros(self.dimensions[0])
        self.bias_summation = np.zeros(1)
        print ("Error: ", round(np.absolute(self.error[0]), 6))

if __name__ == "__main__":                
    neural_network = NeuralNetwork([4,1])

    training_inputs = np.array([[1,0,0,1],
                                [0,1,1,0],
                                [1,1,1,1],
                                [0,0,0,0],
                                [1,1,1,0]])
    training_outputs = np.array([1, 0, 0, 0, 0])
   
    start_time = time.time()
    for i in range (neural_network.epochs):
        print ("\nEpoch", str(i+1) + "/" + str(neural_network.epochs))
        neural_network.train(training_inputs, training_outputs)
    time = time.time() - start_time

    print("\n\n\nTotal Time Used")
    if time > 60:
        print("Minutes: %s\n" % round((time/60),2))
    else:
        print("Seconds: %s\n" % round(time,2))

    for i in range (neural_network.numNewSituation):
        print("\n\nNew Situations: " + str(i+1) + "/" + str(neural_network.numNewSituation))
        A = list(map(int,input("Enter the numbers : ").strip().split()))[:4] 
   
        try:
            result = neural_network.sigmoid(np.dot(neural_network.weights, A) + neural_network.bias)
        except ValueError:
            print("\nValueError, try again")
            continue

        print("\nOutput Data:", result[0])
       
        if result>0.95:
            print("Result: Back Slash")
        else:
            print("Result: Not Back Slash")

    # weights_file = open(r"weights_file", "wb")
    # weights_array = array('f', neural_network.weights)
    # weights_array.tofile(weights_file)
    # weights_file.close

    # bias_file = open(r"bias_file", "wb")
    # bias_array = array('f', neural_network.bias)
    # bias_array.tofile(bias_file)
    # bias_file.close