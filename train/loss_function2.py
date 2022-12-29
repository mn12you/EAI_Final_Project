import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint

import tensorflow as tf
import tensorflow_datasets as tfds
def loss_function(real, pred, pred_teacher, alpha):
  # 這次的 mask 將序列中不等於 0 的位置視為 1，其餘為 0 
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  # 照樣計算所有位置的 cross entropy 但不加總
  # 將序列中最後一個 distribution 取出，並將裡頭值最大的當作模型最新的預測字
  pred_teacher_s = pred_teacher[: , -1:, :]  # (batch_size, 1, vocab_size)

  predicted_id = tf.cast(tf.argmax(pred_teacher, axis=-1), tf.int64)


  loss_ = alpha * loss_object(real, pred) + (1-alpha) * loss_object(predicted_id,pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask  # 只計算非 <pad> 位置的損失 
  
  return tf.reduce_mean(loss_)