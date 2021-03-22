# Return TF.Dataset instead of batch as for TF2

train_data = create_dataset(TRAIN_PATH, True, BATCH_SIZE)
eval_data = create_dataset(EVAL_PATH, False, BATCH_SIZE)
train_data

IMG_SIZE=224
size = (IMG_SIZE, IMG_SIZE)
train_data=train_data.map(lambda image,label:(tf.image.resize(image, size),label))
eval_data=eval_data.map(lambda image,label:(tf.image.resize(image, size),label))