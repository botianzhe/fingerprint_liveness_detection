
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from main import read_data_set
from main import inference
from main import lossy
from main import training
from main import evaluation
# pylint: disable=missing-docstring
import os.path
import sys
import time


from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

IMAGE_SIZE = 64
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 2
PACH_SIZE = 10

def main(_):
  if tf.gfile.Exists('/tmp/tensorflow/finger print/logs/network'):
    tf.gfile.DeleteRecursively('/tmp/tensorflow/finger print/logs/network')
  tf.gfile.MakeDirs('/tmp/tensorflow/finger print/logs/network')
  run_training()

def tfill_feed_dict(data_set, images_pl, labels_pl, keep_pl):

  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(PACH_SIZE, False)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      keep_pl: 0.5,
  }
  return feed_dict


def ffill_feed_dict(data_set, images_pl, labels_pl, keep_pl):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch(PACH_SIZE, False)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        keep_pl: 1.0,
    }
    return feed_dict


def placeholder_inputs(batch_size):

  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         IMAGE_SIZE*IMAGE_SIZE))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def run_training():

  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets = read_data_set()
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(10)
    keep_placeholder = tf.placeholder(tf.float32)

    logits = inference(images_placeholder, keep_placeholder)
    # Add to the Graph the Ops for loss calculation.
    loss = lossy(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, 0.02)

    eval_correct = evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter('/tmp/tensorflow/finger print/logs/network', sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    for step in xrange(40000):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = tfill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder,
                                 keep_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == 2000:
        checkpoint_file = os.path.join('/tmp/tensorflow/finger print/logs/network', 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                keep_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                keep_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                keep_placeholder,
                data_sets.test)

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            keep_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // PACH_SIZE
  num_examples = steps_per_epoch * PACH_SIZE
  for step in xrange(steps_per_epoch):
    feed_dict = ffill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               keep_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
 (num_examples, true_count, precision))



tf.app.run(main=main, argv=[sys.argv[0]])


