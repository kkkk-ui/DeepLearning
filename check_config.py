import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
