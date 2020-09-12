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
for each neuron has
Xs: value of output after activation function
delta: transfer Derivative value
netvalue: netvalue before activation function
```pyhton
def initialize_network(self):
        self.weights_list=[np.random.random((self.n_neurals[i] + 1,self.n_neurals[i + 1])) for i in range(self.n_hidden+1)]
        self.Xs=[]
        self.delta=[]
        self.netvalue=[] 
```
# Forward Propagate 
We can calculate an output from a neural network by propagating an input signal through each layer until the output layer outputs its values.
```python
def forward(self,row):
        #Input Layer Data
        row = np.expand_dims(row, axis=1)
        data_input=np.transpose(row)
        self.Xs.append(data_input)
        XS = np.transpose(row)
        for w in range(len(self.weights_list)):
        # feed forward
            WX = np.dot(XS, self.weights_list[w])
            self.netvalue.append(WX)
            if self.functions=="Sigmoind":
                act_WX = [self.sigmoid(dp) for dp in WX]
            else:
                act_WX = [self.Hyperbolic_tangent(dp) for dp in WX]
            self.Xs.append(act_WX)
            XS= np.insert(act_WX, 0, self.X0, axis=1)

```
# Back Propagate Error
The backpropagation algorithm is named for the way in which weights are trained.
Error is calculated between the expected outputs and the outputs forward propagated from the network. These errors are then propagated backward through the network from the output layer to the hidden layer, assigning blame for the error and updating weights as they go.
```python
def Back(self,expected):
        expected_output=[]
        if expected == 1:
            expected_output = np.array([1, 0, 0])
        elif expected == 0:
            expected_output = np.array([0, 1, 0])
        elif expected == -1:
            expected_output = np.array([0, 0, 1])
        self.get_delta(expected_output)

    # calcualte Back Propagate Error
def get_delta(self,expected_output):
        last_layer_error = (expected_output - self.Xs[-1])
        derivative_value=[]
        if self.functions == "Sigmoind":
            derivative_value = np.array(self.Xs[-1]) * (1 - np.array(self.Xs[-1]))
        else:
            derivative_value = np.array((1 - (np.array(self.Xs[-1]) ** 2)))
        self.delta.append(last_layer_error * derivative_value)
        # Back Propagation
        weights_level = -1
        XS_level = -2
        for backcounter in range(self.n_hidden):
            weights = np.delete(self.weights_list[weights_level], 0, axis=0)
            weights = np.transpose(weights)
            WD = np.dot(self.delta[-1], weights)
            derivative_value=[]
            if self.functions=="Sigmoind":
                derivative_value = np.array(self.Xs[XS_level]) * (1 - np.array(self.Xs[XS_level]))
            else:
                derivative_value = np.array((1 - (np.array(self.Xs[XS_level])**2)))
            derivative_value = np.expand_dims(derivative_value, axis=1)
            derivative_value = derivative_value.reshape(1, len(self.Xs[XS_level][0]))
            self.delta.append(WD * derivative_value)
            weights_level = weights_level - 1
            XS_level = XS_level - 1
```
# Train Network
The network is trained using stochastic gradient descent.
This involves multiple iterations of exposing a training dataset to the network and for each row of data forward propagating the inputs, backpropagating the error and updating the network weights.
```python
 #Update wrights by using calculated values
    def Update_weights(self):
        delta=-1
        Neuron_Values = np.transpose(self.Xs[0])
        Neuron_delta = np.dot(Neuron_Values, self.delta[delta])
        Update_Value = np.array(Neuron_delta) * self.LR
        self.weights_list[0] += (Update_Value)
        delta -= 1
        for w in range(1,len(self.weights_list)):
            Neuron_Value=np.insert(self.Xs[w], 0, self.X0, axis=1)
            Neuron_Value = np.transpose(Neuron_Value)
            Neuron_delta = np.dot(Neuron_Value, self.delta[delta])
            Update_Value = np.array(Neuron_delta)* self.LR
            self.weights_list[w]= (Update_Value) + self.weights_list[w]
            delta-=1
 ```
 # Predict
 Making predictions with a trained neural network is easy enough.
We have already seen how to forward-propagate an input pattern to get an output. This is all we need to do to make a prediction. We can use the output values themselves directly as the probability of a pattern belonging to each output class.
```python
def Get_pred(self,input):
        last_layer = []
        last_layer.append(1)
        data_input = np.transpose(input)
        data_input = np.expand_dims(data_input, axis=1)
        XS = np.transpose(data_input)
        for w in range(len(self.weights_list)):  # feed forward
            WX = np.dot(XS, self.weights_list[w])
            act_WX = [self.sigmoid(dp) for dp in WX]
            XS = np.insert(act_WX, 0, 1, axis=1)
            last_layer[-1] = XS
        return last_layer
```

# desktop application 
Testing neural network by user input by entering features values and predict iris type
