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
