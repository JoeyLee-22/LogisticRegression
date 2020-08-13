#[row][column]
import numpy as np
import time
import copy

class NeuralNetwork():

    width = 35
    numNewSituation = 10
    numTrainingExamples = 5
    input_size = 5
    learning_rate = 0.1

    def __init__(self):
        #enter synaptic weights here for testing, comment above line first
        #self.synaptic_weights = np.array([-13.81418989,8.88580891,-9.58264554,-9.29115806,9.02047906])
        self.synaptic_weights = np.random.rand(self.input_size)
        self.new_synaptic_weights = np.random.rand(self.input_size)
        self.error = np.empty(self.numTrainingExamples)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
   
    def derivative(self, summation, largest_error):
        for j in range (self.input_size):
            for i in range (self.numTrainingExamples):
                if j==0:
                    self.error[i] = self.sigmoid(np.dot(self.synaptic_weights, training_inputs[i]))-training_outputs[i]
                    if self.error[i] > largest_error:
                        largest_error = round(self.error[i], 8)
                summation += self.error[i]*training_inputs[i][j]
                self.new_synaptic_weights[j] = self.synaptic_weights[j] - summation*self.learning_rate
        return largest_error

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations): 
            largest_error = self.derivative(0, 0)
            self.synaptic_weights = np.copy(self.new_synaptic_weights)

            left = self.width * int(iteration/(training_iterations/100)) // 100
            iteration2 = str(training_iterations)
            error = str(largest_error)
            print(str(iteration+1)+"/"+iteration2 + ' ['+'='*left+">"+'.'*((self.width - left)-1)+']' + " - Largest Error: "+error, end = "\r")
        print (iteration2+'/'+iteration2 + " ["+"="*self.width+"]" + " - Largest Error: "+error, end = "\r")

if __name__ == "__main__":
           
    neural_network = NeuralNetwork()
   
    training_inputs = np.array([[1,1,0,0,1],
                                [1,0,1,1,0],
                                [1,1,1,0,1],
                                [1,1,1,1,1],
                                [1,0,0,0,0]])
   
    #change the outputs if change inputs
    training_outputs = np.array([1, 0, 0, 0, 0])
   
    old_synaptic_weights = copy.deepcopy(neural_network.synaptic_weights)

    start_time = time.time()
    print ("\n")
    neural_network.train(training_inputs, training_outputs, 60000)
    time = time.time() - start_time

    print("\n\n\nTotal Time Used")
    if time > 60:
        print("Minutes: %s\n" % round((time/60),2))
    else:
        print("Seconds: %s\n" % round(time,2))
       
    for i in range (neural_network.numNewSituation):
   
        print("\nNew Situations: " + str(i+1) + "/" + str(neural_network.numNewSituation))
       
        A = int(input("Input 1: "))
        B = int(input("Input 2: "))
        C = int(input("Input 3: "))
        D = int(input("Input 4: "))
        E = int(input("Input 5: "))

        print("\nNew Situation: input data = ", A, B, C, D, E)
   
        result = neural_network.sigmoid(np.dot(neural_network.synaptic_weights, [A, B, C, D, E]))

        print("\nOutput Data:", result)
       
        if result>0.95:
            print("Result: Back Slash")
        else:
            print("Result: Not Back Slash")