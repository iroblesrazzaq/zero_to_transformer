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
        # TODO: Initialize weights and biases
        # Use He initialization for weights: np.random.randn(...) * np.sqrt(2/input_size)
        # Initialize biases as zeros
        self.weights = None  # YOUR CODE HERE
        self.biases = None   # YOUR CODE HERE
        
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
        # TODO: Implement forward pass
        # 1. Store the input for later use in backpropagation
        # 2. Compute the linear transformation: y = x * W + b
        # 3. Return the result
        
        # YOUR CODE HERE
        
        return None  # Replace with actual output
    
    def backward(self, dvalues):
        """Backward pass for this layer
        
        Args:
            dvalues (numpy.ndarray): Gradient of the loss with respect to the output of this layer
            
        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input of this layer
        """
        # TODO: Implement backward pass
        # 1. Compute gradient with respect to weights: dL/dW = x^T * dL/dy
        # 2. Compute gradient with respect to biases: dL/db = sum(dL/dy, axis=0)
        # 3. Compute gradient with respect to inputs: dL/dx = dL/dy * W^T
        
        # YOUR CODE HERE
        
        return None  # Replace with gradient with respect to inputs

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
        # TODO: Implement ReLU forward pass
        # 1. Store inputs for backpropagation
        # 2. Apply the ReLU function
        
        # YOUR CODE HERE
        
        return None  # Replace with actual output
    
    def backward(self, dvalues):
        """Backward pass for ReLU activation
        
        Args:
            dvalues (numpy.ndarray): Gradient of the loss with respect to the output
            
        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input
        """
        # TODO: Implement ReLU backward pass
        # Derivative of ReLU is 1 for inputs > 0, 0 otherwise
        
        # YOUR CODE HERE
        
        return None  # Replace with gradient with respect to inputs

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
        # TODO: Implement a numerically stable Softmax
        # 1. Subtract max value from each sample for numerical stability
        # 2. Calculate exponentials (e^x) of inputs
        # 3. Normalize by dividing by sum of exponentials
        
        # YOUR CODE HERE
        
        return None  # Replace with actual output
    
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
    
    def forward(self, y_pred, y_true):
        """Compute cross-entropy loss
        
        Args:
            y_pred (numpy.ndarray): Predicted probabilities from Softmax, shape (batch_size, num_classes)
            y_true (numpy.ndarray): Ground truth values (either as indices or one-hot encoded)
            
        Returns:
            float: Loss value
        """
        # TODO: Implement cross-entropy loss
        # 1. Convert y_true to one-hot if it's not already
        # 2. Clip y_pred to avoid log(0) errors
        # 3. Calculate cross-entropy loss: -sum(y_true * log(y_pred))
        # 4. Average over the batch
        
        # YOUR CODE HERE
        
        return None  # Replace with actual loss
    
    def backward(self):
        """Backward pass for Cross-Entropy loss
        
        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input (y_pred)
        """
        # TODO: Implement backward pass for Cross-Entropy
        # For Softmax + Cross-Entropy, the gradient is simply (y_pred - y_true)
        
        # YOUR CODE HERE
        
        return None  # Replace with gradient

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
        # TODO: Implement forward pass through all layers
        # Pass the input through each layer in sequence
        
        # YOUR CODE HERE
        
        return None  # Replace with output from final layer
    
    def backward(self, y_true):
        """Backward pass through the entire network
        
        Args:
            y_true (numpy.ndarray): Ground truth values
        """
        # TODO: Implement backward pass through all layers
        # 1. Start with gradient from loss function
        # 2. Pass gradient backward through each layer in reverse order
        
        # YOUR CODE HERE
        
    def train_step(self, X, y, learning_rate):
        """Perform one training step (forward, backward, update)
        
        Args:
            X (numpy.ndarray): Input data for this batch
            y (numpy.ndarray): Ground truth for this batch
            learning_rate (float): Learning rate for parameter updates
            
        Returns:
            float: Loss value for this batch
        """
        # TODO: Implement a single training step
        # 1. Perform forward pass
        # 2. Calculate loss
        # 3. Perform backward pass
        # 4. Update parameters
        
        # YOUR CODE HERE
        
        return None  # Replace with loss value
    
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
        # TODO: Implement training loop
        # 1. Loop over epochs
        # 2. Shuffle data for each epoch
        # 3. Split data into mini-batches
        # 4. Perform training step for each batch
        # 5. Optionally perform validation after each epoch
        # 6. Record metrics for plotting
        
        # YOUR CODE HERE
        
        return {}  # Replace with training history
    
    def predict(self, X):
        """Make predictions for input data
        
        Args:
            X (numpy.ndarray): Input data
            
        Returns:
            numpy.ndarray: Predicted class indices
        """
        # TODO: Implement prediction
        # 1. Perform forward pass
        # 2. Take argmax to get predicted class
        
        # YOUR CODE HERE
        
        return None  # Replace with predictions

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
    
    num_neurons = min(weights.shape[0], 25)  # Show at most 25 neurons
    grid_size = int(np.ceil(np.sqrt(num_neurons)))
    
    for i in range(num_neurons):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Reshape weights to input shape
        weight_img = weights[i].reshape(input_shape)
        
        plt.imshow(weight_img, cmap='viridis')
        plt.axis('off')
        plt.title(f'Neuron {i+1}')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    X_train, X_test, y_train, y_test = load_mnist()

    X_sample, y_sample = X_train[:5000], y_train[:5000]
    X_val_sample, y_val_sample = X_test[:1000], y_test[:1000]



    # %%

    model = NeuralNetwork()

    # %%
    model.add(Layer(784, 128))  # Input layer -> Hidden layer
    model.add(ReLU())           # ReLU activation
    model.add(Layer(128, 64))   # Hidden layer -> Hidden layer
    model.add(ReLU())           # ReLU activation
    model.add(Layer(64, 10))    # Hidden layer -> Output layer
    model.add(Softmax())        # Softmax activation


    # %%
    model.set_loss(CrossEntropyLoss())


    # %%
    history = model.train(
        X_sample, y_sample,
        epochs=10,
        batch_size=32,
        learning_rate=0.01,
        X_val=X_val_sample,
        y_val=y_val_sample
    )

    plot_loss_accuracy(history)

    # Visualize weights
    visualize_weights(model)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")


