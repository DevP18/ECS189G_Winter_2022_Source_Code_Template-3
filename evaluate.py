import abc
import torch

# -----------------------------------------------------
# Base evaluate class
class evaluate:

    evaluate_name = None
    evaluate_description = None

    def __init__(self, eName=None, eDescription=None):
        self.evaluate_name = eName
        self.evaluate_description = eDescription

    @abc.abstractmethod
    def evaluate(self):
        pass


# -----------------------------------------------------
# CNN evaluation class
class cnn_evaluate(evaluate):

    def __init__(self):
        super().__init__(
            eName='cnn_evaluate',
            eDescription='CNN evaluation'
        )

    def evaluate(self):
        pass

    def evaluate_model(self, model, test_loader, device):

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                # IMPORTANT FIX for ORL/CIFAR mismatch
                images = images[:, :1, :, :] if images.shape[1] == 3 else images

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total