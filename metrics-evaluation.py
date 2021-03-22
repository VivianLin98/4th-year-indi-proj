axis=AXIS_ML
# Adding epsilon for numerical stability
def precision_c(y_true, y_pred):
    pred = tf.cast(tf.keras.backend.round(y_pred), dtype=tf.float32)
    true = tf.cast(y_true, dtype=tf.float32)
    
    raw_tp = tf.math.count_nonzero(pred * true, axis=axis)
    raw_fp = tf.math.count_nonzero(pred * (true - 1.0), axis=axis)
    
    TP = tf.cast(raw_tp, dtype=tf.float32)
    FP = tf.cast(raw_fp, dtype=tf.float32)
    
    precision = TP / (TP + FP + tf.keras.backend.epsilon())
    return precision
  
def recall_c(y_true, y_pred):
    pred = tf.cast(tf.keras.backend.round(y_pred), tf.float32)
    true = tf.cast(y_true, dtype=tf.float32)
    
    raw_tp = tf.math.count_nonzero(pred * true, axis=axis, dtype=tf.float32)
    raw_fn = tf.math.count_nonzero((pred - 1) * true, axis=axis, dtype=tf.float32)
    
    TP = tf.cast(raw_tp, dtype=tf.float32)
    FN = tf.cast(raw_fn, dtype=tf.float32)
    
    recall = TP / (TP + FN + tf.keras.backend.epsilon())
    return recall

def f1_c(y_true, y_pred):
    precision = precision_c(y_true, y_pred)
    recall = recall_c(y_true, y_pred)
    return 2* ( ( precision * recall ) / ( precision + recall + tf.keras.backend.epsilon() ))