from script_data_loader import *

from dataset import cnn_dataset
from method import cnn_method
from evaluate import cnn_evaluate
from result import cnn_result
from setting import cnn_setting

# -------------------------
# CHANGE THIS ONLY
# -------------------------
dataset_name = 'CIFAR'   # MNIST / ORL / CIFAR

# -------------------------
# LOAD DATASET (FIX)
# -------------------------
if dataset_name == 'MNIST':
    loaded_dataset = load_mnist_dataset()

elif dataset_name == 'ORL':
    loaded_dataset = load_orl_dataset()

elif dataset_name == 'CIFAR':
    loaded_dataset = load_cifar_dataset()

# -------------------------
# MODEL CONFIG
# -------------------------
if dataset_name == 'MNIST':
    input_channel = 1
    num_classes = 10

elif dataset_name == 'ORL':
    input_channel = 1
    num_classes = 40

elif dataset_name == 'CIFAR':
    input_channel = 3
    num_classes = 10

# -------------------------
# CREATE MODULES
# -------------------------
my_dataset = cnn_dataset(loaded_dataset, dataset_name)

my_method = cnn_method(
    input_channel=input_channel,
    num_classes=num_classes
)

my_evaluate = cnn_evaluate()
my_result = cnn_result()
my_setting = cnn_setting()

# -------------------------
# RUN
# -------------------------
my_setting.prepare(my_dataset, my_method, my_result, my_evaluate)

my_setting.print_setup_summary()

my_setting.load_run_save_evaluate()