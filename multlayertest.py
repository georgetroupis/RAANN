"""
Created by George Troupis (2019) for the University of Melbourne subject Real and Artifical Neural Networks.

This program defines a neural network and then runs that neural network, according to assigned parameters.
The neural network is trained on the mnist data set, which consists of 60,000
handwritten numbers depicted as an array of 28x28 pixels and then tested through 
10,000 images. 
The program then prints the performance of the program and how long it took for
it to elapse.
"""

#Imports mnist data set and other necessary modules
from keras.datasets import mnist
import time
import numpy as np 
np.random.seed(1)
import scipy.special
import csv

class neuralNetwork :
    #Initialises the network according to certain parameters
    def __init__(self, inputnodes, hiddennodes, outputnodes, lr, ac_func, bias, leaky_slope, hid_layers):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = lr
        self.activation_function = ac_func
        self.sigmoid = lambda x: scipy.special.expit(x)
        self.bias = bias
        bias_node = 1 if self.bias else 0 
        self.leaky_slope = leaky_slope
        
        #the number of hidden layers
        #the total number of layers will be 2 greater, including the input layer and output layer
        self.layers = hid_layers

        #initialises the weights in a dictionary called d
        self.d = {}
        
        #creation of weights from input layer to first hidden layer
        self.d["w01"] = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes + bias_node))
        
        #creation of weights from hidden layer to hidden layer
        for layer in range(1, self.layers):        
            self.d["w{}{}".format(layer, layer+1)] = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.hnodes + bias_node))
                
        #creation of weights from the last hidden layer to the output layer
        self.d["w{}{}".format(self.layers, self.layers +1)] = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes + bias_node))
    
    #takes an array and passes it through the chosen activation function
    def forward_activation(self, X):     
        if self.activation_function == "sigmoid":      
            return self.sigmoid(X)
        elif self.activation_function == "relu":      
            return np.maximum(0,X)
        elif self.activation_function == "leaky_relu":
            return np.maximum(self.leaky_slope*X,X)

    #takes an array and passes it through the derivative of the chosen activation function
    def grad_activation(self, X):
        if self.activation_function == "sigmoid":
          return X*(1-X) 
        elif self.activation_function == "relu":
          return X>0
        elif self.activation_function == "leaky_relu":
          d=np.zeros_like(X)
          d[X<=0]=self.leaky_slope
          d[X>0]=1
          return d
      
    #the forward pass of the neural network, returning the output 
    def query(self, inputs_list) :
        inputs = np.array(inputs_list, ndmin=2).T
        
        #passes the inputs through the weights and activation function for however many hidden layers there are
        self.h = {}
        for layer in range(self.layers):
            #the inputs to the first layer are just the inputs (ie. the 784 pixels)
            if layer ==0:
                self.h["h0"] = inputs 
            
            #Adds the bias to the inputs
            if self.bias:
                self.h["h{}".format(layer)] = np.append(self.h["h{}".format(layer)], self.bias)
                
                #reshapes the list back into an array
                if layer ==0:
                    self.h["h0"] = np.reshape(self.h["h0"], (self.inodes + self.bias, 1))
                else:
                    self.h["h{}".format(layer)] = np.reshape(self.h["h{}".format(layer)], (self.hnodes +self.bias, 1)) 
                    
            #Passes the data through the weights and the activation function
            hidden_inputs = np.dot(self.d["w{}{}".format(layer, layer +1)], self.h["h{}".format(layer)])
            self.h["h{}".format(layer+1)] = self.forward_activation(hidden_inputs)
            
        #adds a bias to the final hidden layer 
        if self.bias:
            self.h["h{}".format(self.layers)] = np.append(self.h["h{}".format(self.layers)], self.bias)
            self.h["h{}".format(self.layers)] = np.reshape(self.h["h{}".format(self.layers)], (self.hnodes +self.bias, 1))
        
        #passes the data into the output layer, through the sigmoid activation function
        final_inputs = np.dot(self.d["w{}{}".format(self.layers, self.layers +1)], self.h["h{}".format(self.layers)])
        self.h["h{}".format(self.layers+1)] = self.sigmoid(final_inputs)
        
        return self.h["h{}".format(self.layers+1)]
    
    #trains the neural network
    def train(self, inputs_list, targets_list):
        targets = np.array(targets_list, ndmin=2).T
        
        #forward pass of the inputs
        final_outputs = self.query(inputs_list)
        
        #initialises the dictionary of errors
        self.errors = {}
        
        #error of final outputs
        self.errors["e{}".format(self.layers+1)] = targets - final_outputs
        
        #errors of the hidden layers
        for layer in reversed(range(self.layers+1)):
            
            #if bias is present, it is not taken into account on the back 
            #propagation to determine the errors
            if self.bias:
                if layer == self.layers:
                    self.errors["e{}".format(layer)]= np.dot(self.d["w{}{}".format(layer, layer +1)].T, self.errors["e{}".format(layer+1)])
                else:  
                    self.errors["e{}".format(layer)]= np.dot(self.d["w{}{}".format(layer, layer +1)].T, self.errors["e{}".format(layer+1)][:-1])
            else:
                self.errors["e{}".format(layer)]= np.dot(self.d["w{}{}".format(layer, layer +1)].T, self.errors["e{}".format(layer+1)])
            
        #Updating the weights according to the error 
        #If biases are present, they are not taken into account
        if self.bias:
            for layer in reversed(range(self.layers+1)):

                #shape of output layer is different to the rest of the layers
                if layer == self.layers:
                    self.d["w{}{}".format(layer, layer +1)][:,:-1] += self.lr * np.dot(((self.errors["e{}".format(layer+1)]) * self.grad_activation(self.h["h{}".format(layer+1)])), self.h["h{}".format(layer)][:-1].T)

                else:
                    self.d["w{}{}".format(layer, layer +1)][:,:-1] += self.lr * np.dot(((self.errors["e{}".format(layer+1)][:-1]) * self.grad_activation(self.h["h{}".format(layer+1)][:-1])), self.h["h{}".format(layer)][:-1].T)
                
        #If biases are not present, not truncation is needed
        else:
            for layer in reversed(range(self.layers+1)):
                self.d["w{}{}".format(layer, layer +1)] += self.lr * np.dot(((self.errors["e{}".format(layer+1)]) * self.grad_activation(self.h["h{}".format(layer+1)])), (self.h["h{}".format(layer)]).T)

#runs the neural network according to certain parameters
#outputs the performance of the neural network and time taken for it to elapse
def run_nn(hidden_nodes = 200, lr = 0.1, epochs = 1, ac_func = 'sigmoid', bias = None, leaky_slope = 0.01, hid_layers = 1):
    
    input_nodes= 784
    output_nodes= 10
    
    #creates the neural network
    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, lr, ac_func, bias, leaky_slope, hid_layers)
    
    #loads mnist data set
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
      
    #starts the timer
    start = time.time()  
    
    #loops through all of x_train (ie. the 60,000 mnist images), training on them
    for ind, record in enumerate(x_train):
        
        inputs = np.asfarray((record / 255.0 *0.99) +0.01)
        inputs = np.reshape(inputs, (1, input_nodes))
                       
        targets = np.zeros(output_nodes) + 0.01
        targets[y_train[ind]] = 0.99
        
        n.train(inputs, targets) 
    
    #stops the timer and calculates time elapsed
    end = time.time()
    tot_time = end - start
    
    scorecard = []
    
    #tests how well the network is trained against x_test (ie. 10,000) images
    for ind, record in enumerate(x_test):
        correct_label = int(y_test[ind])
        
        inputs = np.asfarray((record / 255.0 * 0.99) + 0.01)
        inputs = np.reshape(inputs, (1, input_nodes))
        
        outputs = n.query(inputs)
        
        label = np.argmax(outputs)
    
        if (label == correct_label):
            scorecard
            scorecard.append(1)
        else:
            scorecard
            scorecard.append(0)
    
    #adds successful results to an array
    scorecard_array = np.asarray(scorecard)
    
    performance= (scorecard_array.sum() /scorecard_array.size) *100
    return performance, tot_time

#starting amount of hidden layers that will be tested
start_hl = 1

#final amount of hidden layers that will be tested
end_hl = 2

#total amount of nodes that will be split across all of the layers
total_nodes = 50

#takes in the above parameters and appends the results to a list
def test_nn(start_hl, end_hl, total_nodes):
    results = []
    
    cur_hl = start_hl

    while cur_hl <= end_hl:
        nodes_p_layer = total_nodes//cur_hl
        print('Starting %d hidden layer --- nodes per layer:' %cur_hl, nodes_p_layer)        
        
        temp = {
                'hidden layer': cur_hl,
                'nodes per layer': nodes_p_layer
                }
        performance, duration = run_nn(hidden_nodes = nodes_p_layer, hid_layers= cur_hl)
        temp['performance'] = performance
        temp['duration'] = duration
        
        results.append(temp)
        cur_hl += 1

    return results


test_results = test_nn(start_hl, end_hl, total_nodes)

#creates a csv file that prints out the data from the test of the neural network
with open('multlayertest.csv', mode = 'w', newline='') as graph_file:
    writer = csv.writer(graph_file, delimiter = ',',quoting = csv.QUOTE_MINIMAL)

    hl_list = ['']
    p_list = ['performance']
    d_list = ['duration']

    for result in test_results:
        npl = result['nodes per layer']
        hl = result['hidden layer']
        hl_list.append('{} nodes ({} layers)'.format(npl, hl))
        p_list.append(result['performance'])
        d_list.append(round(result['duration'], 2))

    writer.writerow(hl_list)
    writer.writerow(p_list)
    writer.writerow(d_list)

print('*done*')
