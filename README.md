# Multilayer_perceptron
### Multi-layer perceptron(Neural Network):
 The Multi-layer perceptron is also referred with names Artificial neural networks or just Neural Networks. This is mostly inspired by the neurons in the brain where it connects multiple other neurons to convert an input to define an output.
Mostly, a linear model which features as inputs in input layer(there can be multiple inputs/neurons)and these inputs are then multiplied with weights and a bias term is added to produce an output. but we also want our MLP to be flexible to be non-linear like the one which we have right now so we do this by adding activation function. The weights are initially randomly assigned to inputs in the forward pass and the output generated will differ from the trained values output, our goal is minimize this and correct the weight in back propagation to find the best fit of weights.

### steps:

* As we are already provided with skeleton code in two files utils.py and multilayer_perceptron.py so in here i have implemented funtions for various activation functions such as:
* identity: if the derivative is true it returns 1 or else it returns same value X.
* sigmoid:if the derivative is true it returns x*(1-x) or else it returns 1/(1+e^(-x). 
* tanh(Hyperbolic Tangent activation function): it usually ranges from [-1,1], if derivative is is True the function returns 1-x^2 else tanh(x)[tanh(x) = (e^x - e^-x)/(e^x + e^-x)
* relu: Also known as rectified linear unit activation function,Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,the derivative of the activation function is returned instead. we implemented 2 helper functions to satisfy the above criteria.
* softmax: This is already provided along with skeleton code.
other functions include:
* cross_entropy: This function is used to compute the entropy loss 
* one_hot encoding: It basically converts a categorical into a numerical value by creating binary valued colums for each category. we crated an output with zeros of size(len(y),no. of categories) and 1's are assigned at each instance when category is found.

* we then implemented the fit function to fit the model to the provided data matrix and targets and to store the corss the cross entropy value:
* steps:
* similiar to knn we do have attributes as below:
* n_hidden:An integer representing the number of neurons in the one hidden layer of the neural network.
* hidden_activation:A string representing the activation function of the hidden layer. 
* n_iterations:An integer representing the number of gradient descent iterations performed by the fit(X, y) method.
* learning_rate:A float representing the learning rate used when updating neural network weights during gradient descent.
* _output_activation:An attribute representing the activation function of the output layer. This is set to the softmax function defined in utils.py.
* _loss_function:An attribute representing the loss function used to compute the loss for each iteration. This is set to the cross_entropy function defined in utils.py.
* _loss_history:A Python list of floats representing the history of the loss function for every 20 iterations that the
algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are complete, the _loss_history list should have length n_iterations / 20.
* _X:A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This is set in the _initialize(X, y) method.
* _y:A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input data used when fitting the model.
* _h_weights: A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer features and the hidden layer neurons.
* _h_bias:A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term and the hidden layer neurons.
* _o_weights:A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer neurons and the output layer neurons.
* _o_bias:A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term neuron and the output layer neurons.
* the fit function include:
* initialize(X, y)
* calculate output from hidden layer, storing input before activation for calculating gradients
* calculate output from last layer, storing input before activation for calculating gradients
* calculate cross entropy loss
loss = self._loss_function(self._y, o_act_output)
* calculate gradients for output layer
* gradient of cross entropy - (y / p) + (1 - y) / (1 - p)
* calculate gradients for hidden layer
* update weights and bias for hidden and output layer
         
* next moved to predict function as shown below:
* h_out = self.hidden_activation(np.dot(X, self._h_weights) + self._h_bias)
* y_out = self._output_activation(np.dot(h_out, self._o_weights) + self._o_bias) amnd return max(Y_out,axis=1)
