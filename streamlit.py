# %%
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckPoint
# %%
(X_train, y_train), (X_test, y_test) = mnist.
