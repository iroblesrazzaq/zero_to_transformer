import unittest
import numpy as np
import sys
import os

# Add the directory containing your neural network implementation to the path
# Uncomment and modify this if your implementation is in a different directory
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your neural network classes - update these imports based on your module structure
from mlp_scratch import Layer, ReLU, Softmax, CrossEntropyLoss, NeuralNetwork

class TestLayerInitialization(unittest.TestCase):
    """Test the initialization of the Layer class."""
    
    def test_weights_shape(self):
        """Test if weights are initialized with the correct shape."""
        input_size, output_size = 5, 3
        layer = Layer(input_size, output_size)
        self.assertEqual(layer.weights.shape, (input_size, output_size), 
                         f"Weight matrix should have shape ({input_size}, {output_size})")
    
    def test_biases_shape(self):
        """Test if biases are initialized with the correct shape."""
        input_size, output_size = 5, 3
        layer = Layer(input_size, output_size)
        self.assertEqual(layer.biases.shape, (1, output_size), 
                         f"Bias vector should have shape (1, {output_size})")
    
    def test_weights_initialization_scale(self):
        """Test if weights are initialized with He initialization scale."""
        # He initialization uses sqrt(2/input_size)
        input_size, output_size = 100, 50
        layer = Layer(input_size, output_size)
        
        # The variance should be approximately 2/input_size for He initialization
        expected_variance = 2/input_size
        actual_variance = np.var(layer.weights)
        
        # Allow for some random variation, but check if it's close
        self.assertAlmostEqual(actual_variance, expected_variance, delta=0.1, 
                              msg="Weights variance should be close to 2/input_size for He initialization")
    
    def test_biases_initialization(self):
        """Test if biases are initialized to zeros."""
        input_size, output_size = 5, 3
        layer = Layer(input_size, output_size)
        self.assertTrue(np.all(layer.biases == 0), 
                       "Biases should be initialized to zeros")

class TestLayerForward(unittest.TestCase):
    """Test the forward pass of the Layer class."""
    
    def test_forward_output_shape(self):
        """Test if forward pass produces output with the correct shape."""
        input_size, output_size = 5, 3
        batch_size = 10
        
        layer = Layer(input_size, output_size)
        inputs = np.random.randn(batch_size, input_size)
        output = layer.forward(inputs)
        
        self.assertEqual(output.shape, (batch_size, output_size), 
                         f"Output should have shape ({batch_size}, {output_size})")
    
    def test_forward_computation(self):
        """Test if forward pass correctly computes the linear transformation."""
        input_size, output_size = 2, 2
        batch_size = 3
        
        # Create a layer with controlled weights and biases
        layer = Layer(input_size, output_size)
        layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.biases = np.array([[0.5, 0.5]])
        
        # Create inputs
        inputs = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        
        # Expected output: inputs @ weights + biases
        expected_output = np.array([
            [4.5, 6.5],  # [1, 1] @ [[1, 2], [3, 4]] + [0.5, 0.5]
            [8.5, 12.5], # [2, 2] @ [[1, 2], [3, 4]] + [0.5, 0.5]
            [12.5, 18.5] # [3, 3] @ [[1, 2], [3, 4]] + [0.5, 0.5]
        ])
        
        # Get actual output
        actual_output = layer.forward(inputs)
        
        # Check if outputs match
        np.testing.assert_array_almost_equal(actual_output, expected_output, 
                                           err_msg="Forward pass computation incorrect")
    
    def test_input_storage(self):
        """Test if the forward pass correctly stores the input for backprop."""
        input_size, output_size = 5, 3
        batch_size = 10
        
        layer = Layer(input_size, output_size)
        inputs = np.random.randn(batch_size, input_size)
        _ = layer.forward(inputs)
        
        self.assertTrue(np.array_equal(layer.input, inputs), 
                        "Layer should store the input during forward pass")
    
    def test_output_storage(self):
        """Test if the forward pass correctly stores the output."""
        input_size, output_size = 5, 3
        batch_size = 10
        
        layer = Layer(input_size, output_size)
        inputs = np.random.randn(batch_size, input_size)
        output = layer.forward(inputs)
        
        self.assertTrue(np.array_equal(layer.output, output), 
                        "Layer should store the output during forward pass")

class TestLayerBackward(unittest.TestCase):
    """Test the backward pass of the Layer class."""
    
    def setUp(self):
        """Set up common test variables."""
        self.input_size = 2
        self.output_size = 3
        self.batch_size = 4
        
        # Create a layer with controlled weights
        self.layer = Layer(self.input_size, self.output_size)
        self.layer.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.layer.biases = np.array([[0.01, 0.02, 0.03]])
        
        # Set input and run forward pass
        self.inputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0]
        ])
        self.layer.forward(self.inputs)
        
        # Create gradient from next layer
        self.dvalues = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2]
        ])
    
    def test_dweights_computation(self):
        """Test if backward pass correctly computes gradients for weights."""
        # Run backward pass
        self.layer.backward(self.dvalues)
        
        # Expected dweights = X.T @ dvalues
        expected_dweights = np.dot(self.inputs.T, self.dvalues)
        
        np.testing.assert_array_almost_equal(self.layer.dweights, expected_dweights, 
                                           err_msg="Gradient for weights computed incorrectly")
    
    def test_dbiases_computation(self):
        """Test if backward pass correctly computes gradients for biases."""
        # Run backward pass
        self.layer.backward(self.dvalues)
        
        # Expected dbiases = sum(dvalues, axis=0, keepdims=True)
        expected_dbiases = np.sum(self.dvalues, axis=0, keepdims=True)
        
        np.testing.assert_array_almost_equal(self.layer.dbiases, expected_dbiases, 
                                           err_msg="Gradient for biases computed incorrectly")
    
    def test_dinputs_computation(self):
        """Test if backward pass correctly computes gradients for inputs."""
        # Run backward pass
        dinputs = self.layer.backward(self.dvalues)
        
        # Expected dinputs = dvalues @ W.T
        expected_dinputs = np.dot(self.dvalues, self.layer.weights.T)
        
        np.testing.assert_array_almost_equal(dinputs, expected_dinputs, 
                                           err_msg="Gradient for inputs computed incorrectly")
    
    def test_dinputs_shape(self):
        """Test if backward pass produces input gradients with the correct shape."""
        # Run backward pass
        dinputs = self.layer.backward(self.dvalues)
        
        # Expected shape of dinputs
        expected_shape = (self.batch_size, self.input_size)
        
        self.assertEqual(dinputs.shape, expected_shape, 
                         f"Input gradients should have shape {expected_shape}")

class TestReLUForward(unittest.TestCase):
    """Test the forward pass of the ReLU activation."""
    
    def test_positive_values(self):
        """Test if ReLU correctly passes through positive values."""
        relu = ReLU()
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        outputs = relu.forward(inputs)
        
        # Expect positive values to remain unchanged
        np.testing.assert_array_equal(outputs, inputs, 
                                    err_msg="ReLU should pass through positive values unchanged")
    
    def test_negative_values(self):
        """Test if ReLU correctly zeros out negative values."""
        relu = ReLU()
        inputs = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        outputs = relu.forward(inputs)
        
        # Expect negative values to become 0
        expected = np.zeros_like(inputs)
        np.testing.assert_array_equal(outputs, expected, 
                                    err_msg="ReLU should convert negative values to zero")
    
    def test_mixed_values(self):
        """Test if ReLU correctly handles a mix of positive and negative values."""
        relu = ReLU()
        inputs = np.array([[1.0, -2.0], [-3.0, 4.0]])
        outputs = relu.forward(inputs)
        
        # Expected output
        expected = np.array([[1.0, 0.0], [0.0, 4.0]])
        
        np.testing.assert_array_equal(outputs, expected, 
                                    err_msg="ReLU should handle mixed positive and negative values correctly")
    
    def test_input_storage(self):
        """Test if ReLU stores inputs correctly for backpropagation."""
        relu = ReLU()
        inputs = np.array([[1.0, -2.0], [-3.0, 4.0]])
        _ = relu.forward(inputs)
        
        self.assertTrue(np.array_equal(relu.input, inputs), 
                        "ReLU should store the input during forward pass")

class TestReLUBackward(unittest.TestCase):
    """Test the backward pass of the ReLU activation."""
    
    def test_gradient_positive_inputs(self):
        """Test if ReLU backward pass correctly propagates gradients for positive inputs."""
        relu = ReLU()
        
        # Set up inputs and run forward pass
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        relu.forward(inputs)
        
        # Set up gradients from next layer
        dvalues = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        # Run backward pass
        dinputs = relu.backward(dvalues)
        
        # For positive inputs, gradient should pass through unchanged
        np.testing.assert_array_equal(dinputs, dvalues, 
                                    err_msg="ReLU backward should pass gradients unchanged for positive inputs")
    
    def test_gradient_negative_inputs(self):
        """Test if ReLU backward pass correctly zeros gradients for negative inputs."""
        relu = ReLU()
        
        # Set up inputs and run forward pass
        inputs = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        relu.forward(inputs)
        
        # Set up gradients from next layer
        dvalues = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        # Run backward pass
        dinputs = relu.backward(dvalues)
        
        # For negative inputs, gradient should be zero
        expected = np.zeros_like(dvalues)
        np.testing.assert_array_equal(dinputs, expected, 
                                    err_msg="ReLU backward should zero out gradients for negative inputs")
    
    def test_gradient_mixed_inputs(self):
        """Test if ReLU backward pass correctly handles a mix of positive and negative inputs."""
        relu = ReLU()
        
        # Set up inputs and run forward pass
        inputs = np.array([[1.0, -2.0], [-3.0, 4.0]])
        relu.forward(inputs)
        
        # Set up gradients from next layer
        dvalues = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        # Run backward pass
        dinputs = relu.backward(dvalues)
        
        # For mixed inputs, gradient should be zeroed only for negative inputs
        expected = np.array([[0.1, 0.0], [0.0, 0.4]])
        np.testing.assert_array_equal(dinputs, expected, 
                                    err_msg="ReLU backward should selectively zero gradients based on input sign")

class TestSoftmaxForward(unittest.TestCase):
    """Test the forward pass of the Softmax activation."""
    
    def test_output_shape(self):
        """Test if Softmax preserves the shape of the input."""
        softmax = Softmax()
        inputs = np.random.randn(10, 5)
        outputs = softmax.forward(inputs)
        
        self.assertEqual(outputs.shape, inputs.shape, 
                        "Softmax should preserve the shape of the input")
    
    def test_output_sum(self):
        """Test if Softmax outputs sum to 1 along the class dimension."""
        softmax = Softmax()
        inputs = np.random.randn(10, 5)
        outputs = softmax.forward(inputs)
        
        # Sum across class dimension (axis=1)
        sums = np.sum(outputs, axis=1)
        
        # Check that each sample sums to approximately 1
        expected = np.ones(10)
        np.testing.assert_array_almost_equal(sums, expected, decimal=6,
                                           err_msg="Softmax outputs should sum to 1 for each sample")
    
    def test_exponential_ratios(self):
        """Test if Softmax correctly calculates probability ratios from exponentials."""
        softmax = Softmax()
        
        # Use simple inputs where we can calculate expected outputs by hand
        inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        outputs = softmax.forward(inputs)
        
        # Manually compute expected outputs
        def manual_softmax(x):
            # Subtract max for numerical stability
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        expected = manual_softmax(inputs)
        
        np.testing.assert_array_almost_equal(outputs, expected, 
                                           err_msg="Softmax calculation incorrect")
    
    def test_numerical_stability(self):
        """Test if Softmax handles large values without numerical overflow."""
        softmax = Softmax()
        
        # Use extremely large values that would cause overflow without stability fix
        inputs = np.array([[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]])
        outputs = softmax.forward(inputs)
        
        # Despite large inputs, output should be valid probabilities
        # For each row, the largest input should have the highest probability
        max_indices = np.argmax(inputs, axis=1)
        output_max_indices = np.argmax(outputs, axis=1)
        
        np.testing.assert_array_equal(max_indices, output_max_indices, 
                                    err_msg="Softmax should assign highest probability to largest input")
        
        # All outputs should be between 0 and 1
        self.assertTrue(np.all(outputs >= 0) and np.all(outputs <= 1), 
                       "Softmax outputs should be valid probabilities (between 0 and 1)")

class TestCrossEntropyLoss(unittest.TestCase):
    """Test the Cross-Entropy loss function."""
    
    def test_perfect_prediction(self):
        """Test loss value when predictions perfectly match targets."""
        loss_fn = CrossEntropyLoss()
        
        # One-hot encoded targets
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Perfect predictions (after softmax)
        y_pred = np.array([[0.9999, 0.0001, 0.0001], 
                           [0.0001, 0.9999, 0.0001], 
                           [0.0001, 0.0001, 0.9999]])
        
        # Calculate loss
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be very small for perfect predictions
        self.assertLess(loss, 0.01, 
                       "Loss should be very small when predictions match targets")
    
    def test_worst_prediction(self):
        """Test loss value when predictions are opposite to targets."""
        loss_fn = CrossEntropyLoss()
        
        # One-hot encoded targets
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Worst predictions (confidence in wrong classes)
        y_pred = np.array([[0.0001, 0.5, 0.4999], 
                           [0.5, 0.0001, 0.4999], 
                           [0.5, 0.4999, 0.0001]])
        
        # Calculate loss
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be high for terrible predictions
        self.assertGreater(loss, 5.0, 
                          "Loss should be high when predictions are opposite to targets")
    
    def test_class_indices_conversion(self):
        """Test if loss function correctly handles targets as class indices."""
        loss_fn = CrossEntropyLoss()
        
        # Targets as class indices
        y_true_indices = np.array([0, 1, 2])
        
        # Predictions
        y_pred = np.array([[0.7, 0.2, 0.1], 
                           [0.1, 0.7, 0.2], 
                           [0.2, 0.1, 0.7]])
        
        # Calculate loss with indices
        loss_indices = loss_fn.forward(y_pred, y_true_indices)
        
        # One-hot encoded targets (equivalent)
        y_true_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Calculate loss with one-hot
        loss_onehot = loss_fn.forward(y_pred, y_true_onehot)
        
        # Both loss values should be the same
        self.assertAlmostEqual(loss_indices, loss_onehot, 
                              msg="Loss calculation should handle both class indices and one-hot encoded targets")
    
    def test_numerical_stability(self):
        """Test if loss function handles extreme probability values."""
        loss_fn = CrossEntropyLoss()
        
        # One-hot encoded targets
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        # Extreme probabilities (almost 0)
        y_pred = np.array([[1e-10, 1-2e-10, 1e-10], 
                           [1e-10, 1e-10, 1-2e-10], 
                           [1-2e-10, 1e-10, 1e-10]])
        
        # This should not cause numerical errors if properly implemented
        loss = loss_fn.forward(y_pred, y_true)
        
        # Loss should be finite (not inf or nan)
        self.assertTrue(np.isfinite(loss), 
                       "Loss should remain finite even with extreme probability values")

class TestCrossEntropyLossBackward(unittest.TestCase):
    """Test the backward pass of the Cross-Entropy loss function."""
    
    def test_gradient_shape(self):
        """Test if backward pass produces gradients with the correct shape."""
        loss_fn = CrossEntropyLoss()
        
        # Set up predictions and targets
        batch_size, num_classes = 5, 3
        y_pred = np.random.random((batch_size, num_classes))
        y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)  # Normalize to valid probabilities
        y_true = np.eye(num_classes)[np.random.randint(0, num_classes, size=batch_size)]
        
        # Forward pass
        loss_fn.forward(y_pred, y_true)
        
        # Backward pass
        gradients = loss_fn.backward()
        
        # Gradients should have same shape as predictions
        self.assertEqual(gradients.shape, y_pred.shape, 
                        "Gradients shape should match predictions shape")
    
    def test_gradient_formula(self):
        """Test if backward pass correctly computes gradients for softmax + cross-entropy."""
        loss_fn = CrossEntropyLoss()
        
        # Set up predictions and targets
        y_pred = np.array([[0.7, 0.2, 0.1], 
                           [0.1, 0.7, 0.2], 
                           [0.2, 0.1, 0.7]])
        y_true = np.array([[1, 0, 0], 
                           [0, 1, 0], 
                           [0, 0, 1]])
        
        # Forward pass
        loss_fn.forward(y_pred, y_true)
        
        # Backward pass
        gradients = loss_fn.backward()
        
        # For softmax + cross-entropy, the gradient is (y_pred - y_true) / batch_size
        expected_gradients = (y_pred - y_true) / y_pred.shape[0]
        
        np.testing.assert_array_almost_equal(gradients, expected_gradients, 
                                           err_msg="Gradient calculation for softmax + cross-entropy is incorrect")

class TestNeuralNetworkForward(unittest.TestCase):
    """Test the forward pass of the entire neural network."""
    
    def create_simple_network(self):
        """Helper to create a simple network with controlled layers."""
        network = NeuralNetwork()
        
        # Layer 1: 2 inputs -> 3 hidden
        layer1 = Layer(2, 3)
        layer1.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        layer1.biases = np.array([[0.01, 0.02, 0.03]])
        
        # ReLU activation
        relu = ReLU()
        
        # Layer 2: 3 hidden -> 2 outputs
        layer2 = Layer(3, 2)
        layer2.weights = np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])
        layer2.biases = np.array([[0.04, 0.05]])
        
        # Softmax activation
        softmax = Softmax()
        
        # Add layers to network
        network.add(layer1)
        network.add(relu)
        network.add(layer2)
        network.add(softmax)
        
        return network
    
    def test_sequential_forward(self):
        """Test if forward pass correctly propagates through all layers."""
        network = self.create_simple_network()
        
        # Create input
        X = np.array([[1.0, 2.0]])
        
        # Run forward pass
        output = network.forward(X)
        
        # Manually compute expected output
        # Layer 1: X @ W1 + b1
        z1 = np.dot(X, network.layers[0].weights) + network.layers[0].biases
        # ReLU
        a1 = np.maximum(0, z1)
        # Layer 2: a1 @ W2 + b2
        z2 = np.dot(a1, network.layers[2].weights) + network.layers[2].biases
        # Softmax
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        expected_output = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        
        np.testing.assert_array_almost_equal(output, expected_output, 
                                           err_msg="Network forward pass computation incorrect")
    
    def test_output_shape(self):
        """Test if network output has the correct shape."""
        network = self.create_simple_network()
        
        # Create batch of inputs
        batch_size = 5
        X = np.random.randn(batch_size, 2)
        
        # Run forward pass
        output = network.forward(X)
        
        # Output should have shape (batch_size, num_classes)
        expected_shape = (batch_size, 2)  # 2 is the output size of our last layer
        
        self.assertEqual(output.shape, expected_shape, 
                        f"Network output should have shape {expected_shape}")

class TestNeuralNetworkBackward(unittest.TestCase):
    """Test the backward pass of the entire neural network."""
    
    def setUp(self):
        """Set up a simple network with loss function."""
        # Create network
        self.network = NeuralNetwork()
        
        # Layer 1: 2 inputs -> 3 hidden
        self.layer1 = Layer(2, 3)
        
        # ReLU activation
        self.relu = ReLU()
        
        # Layer 2: 3 hidden -> 2 outputs
        self.layer2 = Layer(3, 2)
        
        # Softmax activation
        self.softmax = Softmax()
        
        # Add layers to network
        self.network.add(self.layer1)
        self.network.add(self.relu)
        self.network.add(self.layer2)
        self.network.add(self.softmax)
        
        # Loss function
        self.loss_fn = CrossEntropyLoss()
        self.network.set_loss(self.loss_fn)
        
        # Sample data
        self.X = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.y = np.array([0, 1])  # Class indices
    
    def test_backward_gradients_existence(self):
        """Test if backward pass correctly computes gradients for all layers."""
        # Forward pass
        self.network.forward(self.X)
        
        # Loss (assumes y_pred comes from forward pass and is stored)
        y_pred = self.softmax.output
        self.loss_fn.forward(y_pred, self.y)
        
        # Backward pass
        self.network.backward(self.y)
        
        # Check if gradients are computed
        self.assertIsNotNone(self.layer1.dweights, "Layer 1 weight gradients not computed")
        self.assertIsNotNone(self.layer1.dbiases, "Layer 1 bias gradients not computed")
        self.assertIsNotNone(self.layer2.dweights, "Layer 2 weight gradients not computed")
        self.assertIsNotNone(self.layer2.dbiases, "Layer 2 bias gradients not computed")
    
    def test_gradient_shapes(self):
        """Test if backward pass produces gradients with the correct shapes."""
        # Forward pass
        self.network.forward(self.X)
        
        # Loss
        y_pred = self.softmax.output
        self.loss_fn.forward(y_pred, self.y)
        
        # Backward pass
        self.network.backward(self.y)
        
        # Check gradient shapes
        self.assertEqual(self.layer1.dweights.shape, self.layer1.weights.shape, 
                        "Layer 1 weight gradients have incorrect shape")
        self.assertEqual(self.layer1.dbiases.shape, self.layer1.biases.shape, 
                        "Layer 1 bias gradients have incorrect shape")
        self.assertEqual(self.layer2.dweights.shape, self.layer2.weights.shape, 
                        "Layer 2 weight gradients have incorrect shape")
        self.assertEqual(self.layer2.dbiases.shape, self.layer2.biases.shape, 
                        "Layer 2 bias gradients have incorrect shape")

class TestNeuralNetworkTraining(unittest.TestCase):
    """Test the training process of the neural network."""
    
    def setUp(self):
        """Set up a simple network with loss function."""
        # Create network
        self.network = NeuralNetwork()
        
        # Create a simple network
        self.network.add(Layer(2, 4))
        self.network.add(ReLU())
        self.network.add(Layer(4, 3))
        self.network.add(Softmax())
        
        # Set loss function
        self.network.set_loss(CrossEntropyLoss())
        
        # Create synthetic dataset
        np.random.seed(42)  # For reproducibility
        self.X = np.random.randn(100, 2)
        self.y = np.random.randint(0, 3, size=100)  # 3 classes
    
    def test_train_step(self):
        """Test if a single training step works correctly."""
        # Get a small batch
        X_batch = self.X[:10]
        y_batch = self.y[:10]
        
        # Initial prediction - forward pass
        initial_output = self.network.forward(X_batch)
        
        # Calculate initial loss
        initial_loss = self.network.loss_function.forward(initial_output, y_batch)
        
        # Store initial weights
        initial_weights1 = self.network.layers[0].weights.copy()
        initial_weights2 = self.network.layers[2].weights.copy()
        
        # Perform a training step
        learning_rate = 0.01
        loss = self.network.train_step(X_batch, y_batch, learning_rate)
        
        # Check if loss is a finite number
        self.assertTrue(np.isfinite(loss), 
                       "Loss value should be finite after a training step")
        
        # Check if weights are updated
        self.assertFalse(np.array_equal(initial_weights1, self.network.layers[0].weights), 
                        "Layer 1 weights should be updated after a training step")
        self.assertFalse(np.array_equal(initial_weights2, self.network.layers[2].weights), 
                        "Layer 2 weights should be updated after a training step")
    
    def test_train_improves_loss(self):
        """Test if training for multiple steps improves the loss."""
        # Split data into train and validation
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y[:80], self.y[80:]
        
        # Calculate initial loss on validation set
        initial_output = self.network.forward(X_val)
        initial_loss = self.network.loss_function.forward(initial_output, y_val)
        
        # Train for a few epochs with small batch size
        history = self.network.train(
            X_train, y_train,
            epochs=5,
            batch_size=16,
            learning_rate=0.01,
            X_val=X_val,
            y_val=y_val
        )
        
        # Check if training loss decreases
        self.assertLess(history['train_loss'][-1], history['train_loss'][0], 
                       "Training loss should decrease during training")
        
        # Check if validation loss improved
        final_output = self.network.forward(X_val)
        final_loss = self.network.loss_function.forward(final_output, y_val)
        self.assertLess(final_loss, initial_loss, 
                       "Validation loss should improve after training")
    
    def test_history_structure(self):
        """Test if training returns history with the correct structure."""
        # Train for just 2 epochs to save time
        history = self.network.train(
            self.X, self.y,
            epochs=2,
            batch_size=16,
            learning_rate=0.01
        )
        
        # Check if history contains expected keys
        self.assertIn('train_loss', history, "History should track training loss")
        self.assertIn('train_accuracy', history, "History should track training accuracy")
        
        # Check if history values have correct length
        self.assertEqual(len(history['train_loss']), 2, 
                        "History should have one entry per epoch")
        self.assertEqual(len(history['train_accuracy']), 2, 
                        "History should have one entry per epoch")
        
        # Train with validation data
        X_train, X_val = self.X[:80], self.X[80:]
        y_train, y_val = self.y[:80], self.y[80:]
        
        history_with_val = self.network.train(
            X_train, y_train,
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            X_val=X_val,
            y_val=y_val
        )
        
        # Check if validation metrics are tracked
        self.assertIn('val_loss', history_with_val, "History should track validation loss when provided")
        self.assertIn('val_accuracy', history_with_val, "History should track validation accuracy when provided")

class TestNeuralNetworkPredict(unittest.TestCase):
    """Test the prediction functionality of the neural network."""
    
    def setUp(self):
        """Set up a simple network."""
        # Create network
        self.network = NeuralNetwork()
        
        # Create a simple network with controlled weights
        layer1 = Layer(2, 3)
        layer1.weights = np.array([[1.0, -1.0, 0.5], [-0.5, 1.0, -1.0]])
        layer1.biases = np.array([[0.1, 0.1, 0.1]])
        
        relu = ReLU()
        
        layer2 = Layer(3, 2)
        layer2.weights = np.array([[1.0, -1.0], [-1.0, 1.0], [0.5, 0.5]])
        layer2.biases = np.array([[0.1, 0.1]])
        
        softmax = Softmax()
        
        self.network.add(layer1)
        self.network.add(relu)
        self.network.add(layer2)
        self.network.add(softmax)
    
    def test_predict_output_shape(self):
        """Test if predict method returns outputs with the correct shape."""
        # Create batch of inputs
        batch_size = 5
        X = np.random.randn(batch_size, 2)
        
        # Get predictions
        predictions = self.network.predict(X)
        
        # Predictions should be 1D array with length batch_size
        expected_shape = (batch_size,)
        
        self.assertEqual(predictions.shape, expected_shape, 
                        f"Predictions should have shape {expected_shape}")
    
    def test_predict_values(self):
        """Test if predict method returns the class with highest probability."""
        # Create inputs designed to give predictable results with our network
        X = np.array([
            [2.0, 1.0],   # Should predict class 0
            [1.0, 2.0]    # Should predict class 1
        ])
        
        # Run forward pass to get probabilities
        probs = self.network.forward(X)
        
        # Manually get the expected class predictions
        expected_predictions = np.argmax(probs, axis=1)
        
        # Get actual predictions
        predictions = self.network.predict(X)
        
        np.testing.assert_array_equal(predictions, expected_predictions, 
                                     err_msg="Predict should return the class with highest probability")
        
        # Additionally, check if the first example predicts class 0 and second predicts class 1
        self.assertEqual(predictions[0], 0, "First example should predict class 0")
        self.assertEqual(predictions[1], 1, "Second example should predict class 1")

class TestEndToEnd(unittest.TestCase):
    """Test the complete network on a small synthetic dataset."""
    
    def test_mnist_subset(self):
        """Test training and evaluation on a small subset of synthetic MNIST-like data."""
        # Create synthetic data that mimics MNIST format
        np.random.seed(42)
        
        # Create 100 samples, 20 features (simplified from 784)
        X_train = np.random.randn(100, 20)
        
        # 10 classes (digits 0-9)
        y_train = np.random.randint(0, 10, size=100)
        
        # Create a smaller validation set
        X_val = np.random.randn(20, 20)
        y_val = np.random.randint(0, 10, size=20)
        
        # Create network with smaller dimensions for faster testing
        network = NeuralNetwork()
        network.add(Layer(20, 15))  # Input layer -> Hidden layer
        network.add(ReLU())         # ReLU activation
        network.add(Layer(15, 10))  # Hidden layer -> Output layer
        network.add(Softmax())      # Softmax activation
        
        network.set_loss(CrossEntropyLoss())
        
        # Train for just a few epochs
        history = network.train(
            X_train, y_train,
            epochs=5,
            batch_size=16,
            learning_rate=0.01,
            X_val=X_val,
            y_val=y_val
        )
        
        # Check if training metrics are tracked
        self.assertTrue(len(history['train_loss']) == 5, 
                       "Training should complete all epochs")
        self.assertTrue(len(history['val_loss']) == 5, 
                       "Validation should be performed for all epochs")
        
        # Make predictions on validation set
        predictions = network.predict(X_val)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (20,), 
                        "Predictions should match validation set size")
        
        # Predictions should be integers in range [0, 9]
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 9), 
                       "Predictions should be valid class indices")

if __name__ == '__main__':
    unittest.main()