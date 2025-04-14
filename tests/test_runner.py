import unittest
import sys
import os
from colorama import Fore, Style, init

# Initialize colorama
init()

# Import test cases
from neural_network_tests import (
    TestLayerInitialization,
    TestLayerForward,
    TestLayerBackward,
    TestReLUForward,
    TestReLUBackward,
    TestSoftmaxForward,
    TestCrossEntropyLoss,
    TestCrossEntropyLossBackward,
    TestNeuralNetworkForward,
    TestNeuralNetworkBackward,
    TestNeuralNetworkTraining,
    TestNeuralNetworkPredict,
    TestEndToEnd
)

def print_header(text):
    """Print a header with formatting."""
    print("\n" + "=" * 70)
    print(f"  {Fore.CYAN}{text}{Style.RESET_ALL}")
    print("=" * 70)

class CustomTextTestResult(unittest.TextTestResult):
    """Custom test result formatter with colored output."""
    
    def addSuccess(self, test):
        super().addSuccess(test)
        if self.showAll:
            self.stream.writeln(f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.showAll:
            self.stream.writeln(f"{Fore.RED}✗ ERROR{Style.RESET_ALL}")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.showAll:
            self.stream.writeln(f"{Fore.RED}✗ FAIL{Style.RESET_ALL}")
    
    def printErrorList(self, flavor, errors):
        for test, err in errors:
            self.stream.writeln(self.separator1)
            self.stream.writeln(f"{Fore.RED}{flavor}: {self.getDescription(test)}{Style.RESET_ALL}")
            self.stream.writeln(self.separator2)
            self.stream.writeln(f"{err}")

class CustomTextTestRunner(unittest.TextTestRunner):
    """Custom test runner that uses the custom result class."""
    
    def _makeResult(self):
        return CustomTextTestResult(self.stream, self.descriptions, self.verbosity)

def run_test_group(test_class, verbosity=2):
    """Run a specific test class."""
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    result = CustomTextTestRunner(verbosity=verbosity).run(suite)
    return result.wasSuccessful()

def main():
    """Run tests for neural network implementation."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        component = sys.argv[1].lower()
    else:
        component = "all"
    
    # Track overall success
    all_success = True
    
    if component in ["layer", "all"]:
        print_header("TESTING LAYER CLASS")
        init_success = run_test_group(TestLayerInitialization)
        forward_success = run_test_group(TestLayerForward)
        backward_success = run_test_group(TestLayerBackward)
        all_success = all_success and init_success and forward_success and backward_success
    
    if component in ["relu", "all"]:
        print_header("TESTING RELU ACTIVATION")
        relu_forward_success = run_test_group(TestReLUForward)
        relu_backward_success = run_test_group(TestReLUBackward)
        all_success = all_success and relu_forward_success and relu_backward_success
    
    if component in ["softmax", "all"]:
        print_header("TESTING SOFTMAX ACTIVATION")
        softmax_success = run_test_group(TestSoftmaxForward)
        all_success = all_success and softmax_success
    
    if component in ["loss", "all"]:
        print_header("TESTING CROSS-ENTROPY LOSS")
        loss_success = run_test_group(TestCrossEntropyLoss)
        loss_backward_success = run_test_group(TestCrossEntropyLossBackward)
        all_success = all_success and loss_success and loss_backward_success
    
    if component in ["network", "all"]:
        print_header("TESTING NEURAL NETWORK")
        network_forward_success = run_test_group(TestNeuralNetworkForward)
        network_backward_success = run_test_group(TestNeuralNetworkBackward)
        all_success = all_success and network_forward_success and network_backward_success
    
    if component in ["training", "all"]:
        print_header("TESTING NEURAL NETWORK TRAINING")
        training_success = run_test_group(TestNeuralNetworkTraining)
        predict_success = run_test_group(TestNeuralNetworkPredict)
        all_success = all_success and training_success and predict_success
    
    if component in ["endtoend", "all"]:
        print_header("RUNNING END-TO-END TEST")
        endtoend_success = run_test_group(TestEndToEnd)
        all_success = all_success and endtoend_success
    
    # Print summary
    print("\n" + "=" * 70)
    if all_success:
        print(f"{Fore.GREEN}✅ ALL TESTS PASSED!{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ SOME TESTS FAILED{Style.RESET_ALL}")
    print("=" * 70)
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())