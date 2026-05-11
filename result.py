'''
Result module
'''

import abc
import matplotlib.pyplot as plt

# -----------------------------------------------------
# abstract result class
class result:

    result_name = None
    result_description = None

    def __init__(self, rName=None, rDescription=None):

        self.result_name = rName
        self.result_description = rDescription

    @abc.abstractmethod
    def save(self):
        return

    @abc.abstractmethod
    def load(self):
        return



import matplotlib.pyplot as plt

class cnn_result:

    def __init__(self):
        self.result_name = "cnn_result"

    def save_result(self, train_losses, test_accuracies):

        epochs = list(range(1, len(train_losses) + 1))

        # ---------------- LOSS CURVE ----------------
        plt.figure()
        plt.plot(epochs, train_losses)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

        # ---------------- ACCURACY CURVE ----------------
        plt.figure()
        plt.plot(epochs, test_accuracies)
        plt.title("Test Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.show()