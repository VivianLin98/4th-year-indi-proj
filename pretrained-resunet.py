from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
import tensorflow_hub as hub
# Build an input pipeline
def install_pip():
  !pip install tensorflow-gpu==2.2.0

def download_data():
  !curl -O 'https://storage.googleapis.com/big-earth-net/new/tf-data/train-big-earth-net.tfrecords'
  !curl -O 'https://storage.googleapis.com/big-earth-net/new/tf-data/eval-big-earth-net.tfrecords'
  
  !mkdir data
  !mv 'train-big-earth-net.tfrecords' 'eval-big-earth-net.tfrecords' data/

install_pip()
download_data()

# Find the training set & eval set path
TRAIN_PATH = "data/train-big-earth-net.tfrecords"
EVAL_PATH = "data/eval-big-earth-net.tfrecords"

LOGDIR = 'logs'
!mkdir hms

# Set the parameters
SHUFFLE_BUFFER = 70000 # Memory limitation on Google Colab, on HPC switch to 150k
INPUT_SHAPE = (120, 120, 3) # images size
N_CLASSES = 16 # teh class number, bigearthnet labeled classes
LEARNING_RATE = 0.0001
BATCH_SIZE = 1024
EPOCHS = 15
STEPS = 500
VAL_STEPS = STEPS
POS_VALUE = [1]
SEED = 42
AXIS_ML = 1 
"""
1 = 'micro': Calculate metrics globally by counting the total true positives, false negatives and false positives.
None = 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
"""
# Define the ConvBlock
def ConvBlock(X,channel,kernel_size,bn=True):
  x=layers.Conv2D(filters=channel,kernel_size=(kernel_size,kernel_size),strides=(1,1),dilation_rate=(1,1),padding='SAME',kernel_initializer='he_normal')(X)
  if bn:
    x=layers.BatchNormalization()(x)
  x=layers.Activation('relu')(x)

  x=layers.Conv2D(filters=channel,kernel_size=(kernel_size,kernel_size),strides=(1,1),dilation_rate=(1,1),padding='SAME',kernel_initializer='he_normal')(x)
  if bn:
    x=layers.BatchNormalization()(x)
  x=layers.Activation('relu')(x)
  return x

  # For first level -CORNIE first level - has 16 classes
  classes = 16
url = 'https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1'
base_model=hub.load(url,tags="train")

#checkpoint = tf.train.Checkpoint(base_model)
# Save a checkpoint to /tmp/training_checkpoints-{save_counter}. Every time
# checkpoint.save is called, the save counter is increased.
#save_path = checkpoint.save('/tmp/training_checkpoints')
# Restore the checkpointed values to the `model` object.
#checkpoint.restore(save_path)

feature_extractor_layer = hub.KerasLayer(url,input_shape=[224,224,3]) #embed
feature_extractor_layer.trainable = False

inputs = keras.Input(shape=size + (3,))
# build modified U-Net model
model = keras.Sequential([
  # pretrained ResNet50 model
  # output is [None, 2048] num_features = 2048
  
  feature_extractor_layer 
  
])

model.add(tf.keras.layers.Reshape((32,32,2),input_shape=(2048,)))

model.add(layers.Conv2D(8,(3,3), activation="relu", padding='same', input_shape=(512, 512, 4)))
model.add(layers.Conv2D(8,(3,3), activation="relu", padding='same'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(16,(3,3), activation="relu", padding='same'))
model.add(layers.Conv2D(16,(3,3), activation="relu", padding='same'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(32,(3,3), activation="relu", padding='same'))
model.add(layers.Conv2D(32,(3,3), activation="relu", padding='same'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))


model.add(layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same'))
model.add(layers.Conv2D(16,(3,3), activation="relu", padding='same'))
model.add(layers.Conv2D(16,(3,3), activation="relu", padding='same'))

model.add(layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same'))
model.add(layers.Conv2D(8,(3,3), activation="relu", padding='same'))
model.add(layers.Conv2D(classes,(3,3), activation="relu", padding='same'))

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='kullback_leibler_divergence',
    metrics=["accuracy"]
)
#model = Model(input = inputs, output = conv10)
model.summary()

EVALUATION_INTERVAL = 100
#y_pred = model.predict(test_data,batch_size=1, verbose=1)
model.fit(train_data, batch_size=32, validation_data=eval_data, epochs=15, steps_per_epoch=EVALUATION_INTERVAL)