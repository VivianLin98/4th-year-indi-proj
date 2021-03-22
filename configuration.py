# For training
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

from keras import backend as K

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

import wandb
from wandb.keras import WandbCallback

from matplotlib import pyplot as plt

import numpy as np
import itertools
import shutil
import math
from IPython.display import clear_output

tf.keras.backend.clear_session()
tf.__version__

# For BigEarthNet dataset
import matplotlib.pyplot as plt
import numpy as np
import os
#!pip install tensorflow==2.2.0 
import tensorflow as tf

#from tensorflow.keras.preprocessing import image_dataset_from_directory
print("TensorFlow version:" + " "+tf.__version__)