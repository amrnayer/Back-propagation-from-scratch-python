# Back-propagation-from-scratch-python
Implementing neural network back propagation training method by using python from scratch

# Backpropagation Algorithm
The Backpropagation algorithm is a supervised learning method for multilayer feed-forward networks from the field of Artificial Neural Networks.

# Iris dataset
This is the "Iris" dataset. Originally published at UCI Machine Learning Repository: Iris Data Set, this small dataset from 1936 is often used for testing out machine learning algorithms and visualizations (for example, Scatter Plot). Each row of the table represents an iris flower, including its species and dimensions of its botanical parts, sepal and petal, in centimeters.

# Implementation
we can broke it  down into 6 parts:
1. import data
2. Initialize Network.
3. Forward Propagate.
4. Back Propagate Error.
5. Train Network.
6. Predict.

# Import data
Read dataset using pandas and divide dataset into train and test as 70% and 30% respectively

# Initialize Network
Each neuron has a set of weights that need to be maintained. One weight for each input connection and an additional weight for the bias. We will need to store additional properties for a neuron during training, therefore we will use a dictionary to represent each neuron and store properties by names such as ‘weights‘ for the weights.
A network is organized into layers. The input layer is really just a row from our training dataset. The first real layer is the hidden layer. This is followed by the output layer that has one neuron for each class value.
> '''python
def initialize_network(self):
        self.weights_list=[np.random.random((self.n_neurals[i] + 1,self.n_neurals[i + 1])) for i in range(self.n_hidden+1)]
        self.Xs=[]
        self.delta=[]
        self.netvalue=[]
'''      
