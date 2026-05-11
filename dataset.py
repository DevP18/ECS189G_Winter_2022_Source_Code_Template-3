import abc
import numpy as np
import torch
from torch.utils.data import Dataset

class dataset:
    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_description = dDescription

    @abc.abstractmethod
    def load(self):
        pass


class cnn_dataset(dataset):

    def __init__(self, loaded_dataset, dataset_type):
        super().__init__('cnn_dataset', 'CNN image dataset')
        self.loaded_dataset = loaded_dataset
        self.dataset_type = dataset_type

    def get_train_dataset(self):
        return image_dataset(self.loaded_dataset['train'], self.dataset_type)

    def get_test_dataset(self):
        return image_dataset(self.loaded_dataset['test'], self.dataset_type)


class image_dataset(Dataset):

    def __init__(self, data_list, dataset_type):

        self.images = []
        self.labels = []

        for item in data_list:

            image = np.array(item['image'], dtype=np.float32)
            label = int(item['label'])

            # ---------------- IMAGE ----------------
            if dataset_type == 'MNIST':
                image = np.expand_dims(image, axis=0)

            elif dataset_type == 'ORL':
                if image.ndim == 3:
                    image = np.mean(image, axis=2)
                image = np.expand_dims(image, axis=0)
                label -= 1   # ORL FIX (1–40 → 0–39)

            elif dataset_type == 'CIFAR':
                image = np.transpose(image, (2, 0, 1))

            self.images.append(image)
            self.labels.append(label)

        self.images = torch.tensor(np.array(self.images), dtype=torch.float32) / 255.0
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]