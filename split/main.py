import os
import cv2
import numpy as np
from tensorflow.python.framework import dtypes
from six.moves import xrange
from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
import math

IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 2

def extract_train_images():
     train_set_x = []

     for filenames in os.listdir('F:\\data\\BiometrikaTrain\\Live'):
          img1 = cv2.imread("F:/data/BiometrikaTrain/Live/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          flip_1 = cv2.flip(res1, 1)
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          flip_2 = flip_1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          flip_3 = flip_2.tolist()
          train_set_x.append(res1_1_1)
          train_set_x.append(flip_3)

     for filenames in os.listdir('F:\\data\\BiometrikaTrain\\Spoof\\EcoFlex'):
          img1 = cv2.imread("F:/data/BiometrikaTrain/Spoof/EcoFlex/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          flip_1 = cv2.flip(res1, 1)
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          flip_2 = flip_1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          flip_3 = flip_2.tolist()
          train_set_x.append(res1_1_1)
          train_set_x.append(flip_3)

     for filenames in os.listdir('F:\\data\\BiometrikaTrain\\Spoof\\Gelatin'):
          img1 = cv2.imread("F:/data/BiometrikaTrain/Spoof/Gelatin/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          flip_1 = cv2.flip(res1, 1)
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          flip_2 = flip_1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          flip_3 = flip_2.tolist()
          train_set_x.append(res1_1_1)
          train_set_x.append(flip_3)

     for filenames in os.listdir('F:\\data\\BiometrikaTrain\\Spoof\\Latex'):
          img1 = cv2.imread("F:/data/BiometrikaTrain/Spoof/Latex/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          flip_1 = cv2.flip(res1, 1)
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          flip_2 = flip_1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          flip_3 = flip_2.tolist()
          train_set_x.append(res1_1_1)
          train_set_x.append(flip_3)

     for filenames in os.listdir('F:\\data\\BiometrikaTrain\\Spoof\\Silgum'):
          img1 = cv2.imread("F:/data/BiometrikaTrain/Spoof/Silgum/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          flip_1 = cv2.flip(res1, 1)
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          flip_2 = flip_1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          flip_3 = flip_2.tolist()
          train_set_x.append(res1_1_1)
          train_set_x.append(flip_3)

     for filenames in os.listdir('F:\\data\\BiometrikaTrain\\Spoof\\WoodGlue'):
          img1 = cv2.imread("F:/data/BiometrikaTrain/Spoof/WoodGlue/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          flip_1 = cv2.flip(res1, 1)
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          flip_2 = flip_1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          flip_3 = flip_2.tolist()
          train_set_x.append(res1_1_1)
          train_set_x.append(flip_3)

     npdata = np.array(train_set_x)
     npdata = npdata.reshape(4000, IMAGE_SIZE, IMAGE_SIZE, 1)
     return npdata


def generate_train_labels():
     label = np.zeros(4000)
     offset = np.arange(2000)
     label.flat[offset] = 1
     label = label.astype(int)

     return label


def extract_eval_image():
     eval_set = []

     for filenames in os.listdir('F:\\data\\BiometrikaTest\\Live'):
          img1 = cv2.imread("F:/data/BiometrikaTest/Live/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)


     for filenames in os.listdir('F:\\data\\BiometrikaTest\\Spoof\\EcoFlex'):
          img1 = cv2.imread("F:/data/BiometrikaTest/Spoof/EcoFlex/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)

     for filenames in os.listdir('F:\\data\\BiometrikaTest\\Spoof\\Gelatin'):
          img1 = cv2.imread("F:/data/BiometrikaTest/Spoof/Gelatin/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)

     for filenames in os.listdir('F:\\data\\BiometrikaTest\\Spoof\\Latex'):
          img1 = cv2.imread("F:/data/BiometrikaTest/Spoof/Latex/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)

     for filenames in os.listdir('F:\\data\\BiometrikaTest\\Spoof\\Silgum'):
          img1 = cv2.imread("F:/data/BiometrikaTest/Spoof/Silgum/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)

     for filenames in os.listdir('F:\\data\\BiometrikaTest\\Spoof\\WoodGlue'):
          img1 = cv2.imread("F:/data/BiometrikaTest/Spoof/WoodGlue/"+filenames, cv2.IMREAD_GRAYSCALE)
          res1 = cv2.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
          res1_1 = res1.reshape(1, IMAGE_PIXELS) / 255
          res1_1_1 = res1_1.tolist()
          eval_set.append(res1_1_1)

     npdata = np.array(eval_set)
     npdata = npdata.reshape(2000, IMAGE_SIZE, IMAGE_SIZE, 1)
     return npdata

def generate_eval_label():
     label = np.zeros(2000)
     offset = np.arange(1000)
     label.flat[offset] = 1
     label = label.astype(int)

     return label

def read_data_set(validation_size=100, dtype=dtypes.float32,
                   reshape=True):
     train_image = extract_train_images()
     train_label = generate_train_labels()

     eval_image = extract_eval_image()
     eval_label = generate_eval_label()

     validation_images = train_image[:validation_size]
     validation_labels = train_label[:validation_size]

     train_image = train_image[validation_size:]
     train_label = train_label[validation_size:]

     train = DataSet(train_image, train_label, dtype=dtype, reshape=reshape)
     validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
     test = DataSet(eval_image, eval_label, dtype=dtype, reshape=reshape)

     return base.Datasets(train=train, validation=validation, test=test)



class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:

        images = images.astype(np.float32)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 1
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, unit):
  initial = tf.truncated_normal(shape,
                            stddev=math.sqrt(2.0 / float(unit)))
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def inference(images, keep):

  input1 = tf.reshape(images, [-1, 64, 64, 1])

  filter1 = weight_variable([3, 3, 1, 64], 3 * 3 * 1)
  bias1 = bias_variable([64])

  h_conv1 = tf.nn.relu(conv2d(input1, filter1) + bias1)
  input2 = max_pool_2x2(h_conv1)

  filter2 = weight_variable([3, 3, 64, 128], 3 * 3 * 64)
  bias2 = bias_variable([128])

  h_conv2 = tf.nn.relu(conv2d(input2, filter2) + bias2)
  input3 = max_pool_2x2(h_conv2)

  filter3 = weight_variable([3, 3, 128, 256], 3 * 3 * 128)
  bias3 = bias_variable([256])

  h_conv3 = tf.nn.relu(conv2d(input3, filter3) + bias3)

  filter4 = weight_variable([3, 3, 256, 128], 3 * 3 * 256)
  bias4 = bias_variable([128])

  h_conv4 = tf.nn.relu(conv2d(h_conv3, filter4) + bias4)
  input5 = max_pool_2x2(h_conv4)

  input5_flat = tf.reshape(input5, [-1, 8 * 8 * 128])


  # Linear
  with tf.name_scope('full_connect1'):
    weights = tf.Variable(
        tf.truncated_normal([8 * 8 * 128, 1024],
                            stddev=1.0 / math.sqrt(float(8 * 8 * 128))),
        name='weights')
    biases = tf.Variable(tf.zeros([1024]),
                         name='biases')
    tinput6 = tf.matmul(input5_flat, weights) + biases
    input6 = tf.nn.dropout(tinput6, keep)

  with tf.name_scope('full_connect2'):
    weights = tf.Variable(
        tf.truncated_normal([1024, 512],
                            stddev=1.0 / math.sqrt(float(1024))),
        name='weights')
    biases = tf.Variable(tf.zeros([512]),
                         name='biases')
    tinput7 = tf.matmul(input6, weights) + biases
    input7 = tf.nn.dropout(tinput7, keep)

  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([512, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(512))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(input7, weights) + biases

  return logits


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def lossy(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
