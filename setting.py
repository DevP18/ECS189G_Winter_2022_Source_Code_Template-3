import abc
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class setting:

    def __init__(self, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_size = 64
        self.learning_rate = 0.001
        self.epochs = 20

    def prepare(self, dataset, method, result, evaluate):
        self.dataset = dataset
        self.method = method
        self.result = result
        self.evaluate = evaluate

    def print_setup_summary(self):
        print(
            "dataset:", self.dataset.dataset_name,
            ", method:", self.method.method_name,
            ", setting:", self.setting_name,
            ", result:", self.result.result_name,
            ", evaluation:", self.evaluate.evaluate_name
        )

    def load_run_save_evaluate(self):
        pass


class cnn_setting(setting):

    def __init__(self):
        super().__init__('cnn_setting', 'CNN experiment')

    def load_run_save_evaluate(self):

        train_loader = DataLoader(
            self.dataset.get_train_dataset(),
            batch_size=self.batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            self.dataset.get_test_dataset(),
            batch_size=self.batch_size,
            shuffle=False
        )

        model = self.method.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

      
        train_losses = []
        test_accuracies = []

        for epoch in range(self.epochs):

            model.train()
            total_loss = 0

            for images, labels in train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            acc = self.evaluate.evaluate_model(model, test_loader, self.device)

  
            train_losses.append(avg_loss)
            test_accuracies.append(acc)

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Accuracy: {acc:.2f}%"
            )

   
        self.result.save_result(train_losses, test_accuracies)