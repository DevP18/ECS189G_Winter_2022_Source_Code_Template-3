import pickle

from dataset import cnn_dataset
from method import cnn_method
from evaluate import cnn_evaluate
from result import cnn_result
from setting import cnn_setting

# -----------------------------------------------------
# LOAD DATASET

# choose dataset:
# 'MNIST'
# 'ORL'
# 'CIFAR'

dataset_name = 'ORL'

f = open(dataset_name, 'rb')

loaded_dataset = pickle.load(f)

f.close()

# -----------------------------------------------------
# DATASET SETTINGS

if dataset_name == 'MNIST':

    input_channel = 1
    num_classes = 10

elif dataset_name == 'ORL':

    input_channel = 1
    num_classes = 40

elif dataset_name == 'CIFAR':

    input_channel = 3
    num_classes = 10

# -----------------------------------------------------
# CREATE MODULES
dataset_name = 'ORL'   # or MNIST or CIFAR

my_dataset = cnn_dataset(
    loaded_dataset,
    dataset_name
)

my_method = cnn_method(
    input_channel=input_channel,
    num_classes=num_classes
)

my_evaluate = cnn_evaluate()

my_result = cnn_result()

my_setting = cnn_setting()

# -----------------------------------------------------
# PREPARE

my_setting.prepare(
    my_dataset,
    my_method,
    my_result,
    my_evaluate
)

# -----------------------------------------------------
# PRINT SUMMARY

my_setting.print_setup_summary()

# -----------------------------------------------------
# RUN EXPERIMENT

my_setting.load_run_save_evaluate()