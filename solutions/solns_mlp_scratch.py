# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# %%
class Layer:
    """Base class for neural network layers"""
    
    def __init__(self, input_size, output_size):
        """Initialize a layer with random weights and zero biases
        
        Args:
            input_size (int): Size of the input to this layer
            output_size (int): Size of the output from this layer
        """
        # He initialization for weights
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2/input_size)
        self.biases = np.zeros((1, output_size))
        
        # For storing values needed in backpropagation
        self.input = None
        self.output = None
        
        # For storing gradients
        self.dweights = None
        self.dbiases = None
    
    def forward(self, inputs):
        """Forward pass for this layer
        
        Args:
            inputs (numpy.ndarray): Inputs to this layer, shape (batch_size, input_size)
            
        Returns:
            numpy.ndarray: Outputs from this layer, shape (batch_size, output_size)
        """
        # Store the input for later use in backpropagation
        self.input = inputs
        
        # Compute the linear transformation: y = x * W + b
        self.output = np.dot(inputs, self.weights) + self.biases
        
        return self.output
    
    def backward(self, dvalues):
        """Backward pass for this layer
        
        Args:
            dvalues (numpy.ndarray): Gradient of the loss with respect to the output of this layer
            
        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input of this layer
        """
        # Compute gradient with respect to weights: dL/dW = x^T * dL/dy
        self.dweights = np.dot(self.input.T, dvalues)
        
        # Compute gradient with respect to biases: dL/db = sum(dL/dy, axis=0)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Compute gradient with respect to inputs: dL/dx = dL/dy * W^T
        dinputs = np.dot(dvalues, self.weights.T)
        
        return dinputs

# %%
class ReLU:
    """ReLU activation function layer"""
    
    def __init__(self):
        """Initialize ReLU layer"""
        self.input = None
        self.output = None
    
    def forward(self, inputs):
        """Apply ReLU activation: max(0, x)
        
        Args:
            inputs (numpy.ndarray): Input values
            
        Returns:
            numpy.ndarray: Outputs after applying ReLU
        """
        # Store inputs for backpropagation
        self.input = inputs
        
        # Apply the ReLU function
        self.output = np.maximum(0, inputs)
        
        return self.output
    
    def backward(self, dvalues):
        """Backward pass for ReLU activation
        
        Args:
            dvalues (numpy.ndarray): Gradient of the loss with respect to the output
            
        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input
        """
        # Make a copy of the gradient
        dinputs = dvalues.copy()
        
        # Apply ReLU derivative: 1 for inputs > 0, 0 otherwise
        dinputs[self.input <= 0] = 0
        
        return dinputs

# %%
class Softmax:
    """Softmax activation for output layer"""
    
    def __init__(self):
        """Initialize Softmax layer"""
        self.input = None
        self.output = None
    
    def forward(self, inputs):
        """Apply Softmax activation
        
        Args:
            inputs (numpy.ndarray): Input values
            
        Returns:
            numpy.ndarray: Probability distribution (sums to 1 along axis 1)
        """
        # Store input for backpropagation
        self.input = inputs
        
        # Subtract max value from each sample for numerical stability
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize by dividing by sum of exponentials
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities
        
        return self.output
    
    def backward(self, dvalues):
        """Backward pass for Softmax activation
        
        Args:
            dvalues (numpy.ndarray): Gradient of the loss with respect to the output
            
        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input
        """
        # Note: For this assignment, we'll handle Softmax + Cross-Entropy together 
        # in the loss function for simplicity
        return dvalues

# %%
class CrossEntropyLoss:
    """Cross-entropy loss for classification"""
    
    def __init__(self):
        """Initialize Cross-Entropy loss"""
        self.output = None
        self.y_true = None
        self.y_pred = None
    
    def forward(self, y_pred, y_true):
        """Compute cross-entropy loss
        
        Args:
            y_pred (numpy.ndarray): Predicted probabilities from Softmax, shape (batch_size, num_classes)
            y_true (numpy.ndarray): Ground truth values (either as indices or one-hot encoded)
            
        Returns:
            float: Loss value
        """
        # Store for backward pass
        self.y_pred = y_pred
        
        # Get number of samples
        batch_size = y_pred.shape[0]
        
        # Convert y_true to one-hot if it's not already (if it's 1D array of class indices)
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            self.y_true = np.eye(y_pred.shape[1])[y_true.reshape(-1).astype(int)]
        else:
            self.y_true = y_true
            
        # Clip y_pred to avoid log(0) errors
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate cross-entropy loss
        negative_log_likelihoods = -np.sum(self.y_true * np.log(y_pred_clipped), axis=1)
        
        # Average over the batch
        loss = np.mean(negative_log_likelihoods)
        
        return loss
    
    def backward(self):
        """Backward pass for Cross-Entropy loss
        
        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input (y_pred)
        """
        # For Softmax + Cross-Entropy, the gradient is simply (y_pred - y_true)
        # Get number of samples
        batch_size = self.y_pred.shape[0]
        
        # Calculate gradient
        dinputs = self.y_pred - self.y_true
        
        # Normalize gradient
        dinputs = dinputs / batch_size
        
        return dinputs



# %%
class NeuralNetwork:
    """Neural network with arbitrary layer structure"""
    
    def __init__(self):
        """Initialize an empty neural network"""
        self.layers = []
        self.loss_function = None
    
    def add(self, layer):
        """Add a layer to the network
        
        Args:
            layer: Layer object to add to the network
        """
        self.layers.append(layer)
    
    def set_loss(self, loss_function):
        """Set the loss function for the network
        
        Args:
            loss_function: Loss function object
        """
        self.loss_function = loss_function
    
    def forward(self, X):
        """Forward pass through the entire network
        
        Args:
            X (numpy.ndarray): Input data, shape (batch_size, input_size)
            
        Returns:
            numpy.ndarray: Output predictions
        """
        # Pass the input through each layer in sequence
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def backward(self, y_true):
        """Backward pass through the entire network
        
        Args:
            y_true (numpy.ndarray): Ground truth values
        """
        # Start with gradient from loss function
        grad = self.loss_function.backward()
        
        # Pass gradient backward through each layer in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def train_step(self, X, y, learning_rate):
        """Perform one training step (forward, backward, update)
        
        Args:
            X (numpy.ndarray): Input data for this batch
            y (numpy.ndarray): Ground truth for this batch
            learning_rate (float): Learning rate for parameter updates
            
        Returns:
            float: Loss value for this batch
        """
        # Perform forward pass
        y_pred = self.forward(X)
        
        # Calculate loss
        loss = self.loss_function.forward(y_pred, y)
        
        # Perform backward pass
        self.backward(y)
        
        # Update parameters
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= learning_rate * layer.dweights
                layer.biases -= learning_rate * layer.dbiases
        
        return loss
    
    def train(self, X, y, epochs, batch_size, learning_rate, X_val=None, y_val=None):
        """Train the network
        
        Args:
            X (numpy.ndarray): Training data
            y (numpy.ndarray): Training labels
            epochs (int): Number of epochs to train for
            batch_size (int): Size of each mini-batch
            learning_rate (float): Learning rate for parameter updates
            X_val (numpy.ndarray, optional): Validation data
            y_val (numpy.ndarray, optional): Validation labels
            
        Returns:
            dict: Training history (loss, accuracy, etc.)
        """
        history = {
            'train_loss': [],
            'train_accuracy': []
        }

        X = np.array(X)
        y = np.array(y)
        
        if X_val is not None:
            X_val = np.array(X_val)
        if y_val is not None:
            y_val = np.array(y_val)

        
        if X_val is not None and y_val is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []
        
        num_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data for this epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Variables to track epoch statistics
            epoch_loss = 0
            y_pred_all = []
            
            # Process mini-batches
            for i in range(0, num_samples, batch_size):
                # Get mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Perform training step
                batch_loss = self.train_step(X_batch, y_batch, learning_rate)
                epoch_loss += batch_loss * X_batch.shape[0]
                
                # Store predictions for accuracy calculation
                y_pred_batch = self.predict(X_batch)
                y_pred_all.extend(y_pred_batch)
            
            # Calculate epoch metrics
            epoch_loss /= num_samples
            y_pred_all = np.array(y_pred_all)
            accuracy = accuracy_score(y_shuffled[:len(y_pred_all)], y_pred_all)
            
            # Store metrics
            history['train_loss'].append(epoch_loss)
            history['train_accuracy'].append(accuracy)
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}", end="")
            
            # Validation if data provided
            if X_val is not None and y_val is not None:
                # Get predictions and loss on validation set
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function.forward(y_val_pred, y_val)
                
                # Calculate accuracy
                val_accuracy = accuracy_score(y_val, self.predict(X_val))
                
                # Store metrics
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                print(f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}", end="")
            
            print()  # New line
        
        return history
    
    def predict(self, X):
        """Make predictions for input data
        
        Args:
            X (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Predicted class indices
        """
        # Perform forward pass
        y_pred = self.forward(X)
        
        # Take argmax to get predicted class
        predictions = np.argmax(y_pred, axis=1)
        
        return predictions

def load_mnist():
    """Load MNIST dataset
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load data from sklearn-compatible source
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Normalize pixel values
    X = X / 255.0
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def plot_loss_accuracy(history):
    """Plot the loss and accuracy curves
    
    Args:
        history (dict): Training history containing loss and accuracy values
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_weights(model, input_shape=(28, 28)):
    """Visualize the weights of the first layer
    
    Args:
        model (NeuralNetwork): Trained neural network
        input_shape (tuple): Shape of input images (height, width)
    """
    # Extract weights from the first layer
    weights = model.layers[0].weights
    
    # Plot weights
    plt.figure(figsize=(15, 15))
    
    num_neurons = min(weights.shape[1], 25)  # Show at most 25 neurons
    grid_size = int(np.ceil(np.sqrt(num_neurons)))
    
    for i in range(num_neurons):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Reshape weights to input shape
        weight_img = weights[:, i].reshape(input_shape)
        
        plt.imshow(weight_img, cmap='viridis')
        plt.axis('off')
        plt.title(f'Neuron {i+1}')
    
    plt.tight_layout()
    plt.show()

# %%
def load_mnist():
    """Load MNIST dataset
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load data from sklearn-compatible source
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Normalize pixel values
    X = X / 255.0
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def plot_loss_accuracy(history):
    """Plot the loss and accuracy curves
    
    Args:
        history (dict): Training history containing loss and accuracy values
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_weights(model, input_shape=(28, 28)):
    """Visualize the weights of the first layer
    
    Args:
        model (NeuralNetwork): Trained neural network
        input_shape (tuple): Shape of input images (height, width)
    """
    # Extract weights from the first layer
    weights = model.layers[0].weights
    
    # Plot weights
    plt.figure(figsize=(15, 15))
    
    num_neurons = min(weights.shape[1], 25)  # Show at most 25 neurons
    grid_size = int(np.ceil(np.sqrt(num_neurons)))
    
    for i in range(num_neurons):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Reshape weights to input shape
        weight_img = weights[:, i].reshape(input_shape)
        
        plt.imshow(weight_img, cmap='viridis')
        plt.axis('off')
        plt.title(f'Neuron {i+1}')
    
    plt.tight_layout()
    plt.show()
