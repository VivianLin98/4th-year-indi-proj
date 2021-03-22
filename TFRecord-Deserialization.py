def _parse_function(serialized):
    features = \
    {
        'shape': tf.io.VarLenFeature(tf.int64),
        'image_raw': tf.io.FixedLenFeature((), tf.string),
        'label_raw': tf.io.FixedLenFeature((N_CLASSES), tf.int64)
    }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)

    shape = parsed_example['shape']
    image_raw = parsed_example['image_raw']
    label = parsed_example['label_raw']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.io.decode_raw(image_raw, tf.uint16)
    image = tf.reshape(image, INPUT_SHAPE) # FUTURE, Change this to get the sample shape from parsed shape
    image.set_shape(INPUT_SHAPE)

    # Logits and labels must have the same type and shape
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    
    image = tf.identity(image, name="image")
    label = tf.identity(label, name="label")

    return (image, label) # image, tf.sparse_tensor_to_dense(label) If using VarLenFeature
      
def create_dataset(filenames, perform_shuffle=False, batch_size=1):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    
    if perform_shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
    dataset = dataset.repeat()  
    dataset = dataset.batch(batch_size)  # Batch size to use
    
    return dataset

    raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)