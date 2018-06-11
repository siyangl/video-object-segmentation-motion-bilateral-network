import os
import numpy as np
import tensorflow as tf
import logging

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim

from data import davis_flow_dataset


flags.DEFINE_string('dataset', 'Davis', 'Dataset used for training')
flags.DEFINE_string('split_train', 'train', 'Dataset split')
flags.DEFINE_integer('num_classes', 2, 'Number of classes')
flags.DEFINE_string('data_dir', './FlowExamples', 'Dir to TFExamples.')

flags.DEFINE_float('sigma', 3., 'Initial sigma for the Gaussian filter')
flags.DEFINE_integer('kernel_size', 21, 'kernel size')  # for optical flow
flags.DEFINE_integer('loc_kernel_size', 37, 'kernel size of loc dim')

flags.DEFINE_string('checkpoint_dir', None,
                    'Directory to write checkpoints to.')

flags.DEFINE_float('momentum', 0.9, 'Momentum value for training.')
flags.DEFINE_integer('save_checkpoint_secs', 3600,
                     'Sec interval for checkpoint saving.')

# Learning parameters
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_string('learning_policy', 'step', 'learning policy')
flags.DEFINE_float('initial_learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('steps_per_decay', 4000,
                     'Steps per decay for Step Decay')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay factor.')
flags.DEFINE_integer('total_num_steps', int(1e5), 'Number of steps to take.')
flags.DEFINE_float('learning_rate_decay', 0.9, 'Learning rate decay factor'
                   ' for Step Decay schedule.')
# These two initial distribution does not make much difference.
flags.DEFINE_string('init_weight', 'Gaussian', 'Gaussian or uniform')
flags.DEFINE_integer('random_select_k_bg', 50000,
                     'Random select k pixels from low objectness region for data augmentation')
FLAGS = flags.FLAGS


def create_2d_gaussian_kernel(kernel_size, sigma):
  half_window = (kernel_size-1)//2
  h_feat = np.arange(-half_window, half_window+1)
  w_feat = np.arange(-half_window, half_window+1)
  hw_feat = np.array(np.meshgrid(h_feat, w_feat, indexing='ij'))
  kernel = np.sum(0.5*(hw_feat/sigma)**2, axis=0)
  kernel = np.exp(-kernel)
  kernel = kernel/np.sum(kernel)
  return kernel

def create_1d_gaussian_kernel(kernel_size, sigma):
  half_window = (kernel_size - 1) // 2
  h_feat = np.arange(-half_window, half_window + 1)
  kernel = np.exp(-0.5*(h_feat/sigma)**2)
  kernel = kernel/np.sum(kernel)
  return kernel


def create_2d_uniform_kernel(kernel_size):
  kernel = np.ones((kernel_size, kernel_size))
  kernel /= np.sum(kernel)
  return kernel

def create_1d_uniform_kernel(kernel_size):
  kernel = np.ones(kernel_size)
  kernel /= np.sum(kernel)
  return kernel


def l2_regularizer(weight=1.0, scope=None):
  """Define a L2 regularizer.

  Args:
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.

  Returns:
    a regularizer function.
  """
  def regularizer(tensor):
    with tf.op_scope([tensor], scope, 'L2Regularizer'):
      l2_weight = tf.convert_to_tensor(weight,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight')
      return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
  return regularizer


def main(_):
  """Sets up deeplab training."""
  # Data_info and param_info of the experiments.


  logging.info(FLAGS.checkpoint_dir)
  if FLAGS.checkpoint_dir is None:
    print("Checkpoint dir is required.")
    return
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  flow_grid_size=40
  grid_size=18
  frame_shape = (480, 854)
  dataset = davis_flow_dataset.get_split(FLAGS.split_train, FLAGS.data_dir)
  (labels, lattice, slice_index, objectness, sequence_name,
   timestep) = davis_flow_dataset.provide_data_augmented(
    dataset, shape=frame_shape, shuffle=True, num_epochs=None,
    batch_size=FLAGS.batch_size, random_select_k=FLAGS.random_select_k_bg)

  if FLAGS.init_weight == 'Gaussian':
    flow_init = create_1d_gaussian_kernel(FLAGS.kernel_size, FLAGS.sigma)
  elif FLAGS.init_weight == 'Uniform':
    flow_init = create_1d_uniform_kernel(FLAGS.kernel_size)
  else:
    raise ValueError("Unrecognized init kernel.")

  loc_init = create_1d_uniform_kernel(kernel_size=FLAGS.loc_kernel_size)

  # filters = tf.Variable(init, dtype=tf.float32)
  flow_weights_dx = tf.contrib.framework.variable('weights/dx',
                                                  shape=flow_init.shape,
                                                  # initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                  initializer=tf.constant_initializer(flow_init),
                                                  regularizer=l2_regularizer(FLAGS.weight_decay),
                                                  trainable=True)
  flow_weights_dy = tf.contrib.framework.variable('weights/dy',
                                                  shape=flow_init.shape,
                                                  # initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                  initializer=tf.constant_initializer(flow_init),
                                                  regularizer=l2_regularizer(FLAGS.weight_decay),
                                                  trainable=True)

  weights_x = tf.contrib.framework.variable('weights/x',
                                            shape=loc_init.shape,
                                            # initializer=tf.truncated_normal_initializer(stddev=0.01),
                                            initializer=tf.constant_initializer(loc_init),
                                            regularizer=l2_regularizer(FLAGS.weight_decay),
                                            trainable=True)
  weights_y = tf.contrib.framework.variable('weights/y',
                                            shape=loc_init.shape,
                                            # initializer=tf.truncated_normal_initializer(stddev=0.01),
                                            initializer=tf.constant_initializer(loc_init),
                                            regularizer=l2_regularizer(FLAGS.weight_decay),
                                            trainable=True)

  lattice = tf.expand_dims(lattice, 5)  # Dummy channel dimension
  weights = tf.matmul(tf.expand_dims(flow_weights_dx, 1),
                      tf.expand_dims(flow_weights_dy, 0))
  loc_weights = tf.matmul(tf.expand_dims(weights_x, 1),
                          tf.expand_dims(weights_y, 0))

  # The 4-D convolution is done by 4 consecutive 1-D convolution
  # The input tensor is 6s-D (batch, x, y, dx, dy, 1).
  # Since there is no API handling batched 5-D tensors (6-D),
  # we do some dirty tricks by transposing and reshaping, then use the TF API.
  # There is no bias term
  filters_y = tf.reshape(weights_y, [-1, 1, 1, 1, 1])
  filters_x = tf.reshape(weights_x, [-1, 1, 1, 1, 1])
  filters_dx = tf.reshape(flow_weights_dx, [1, -1, 1, 1, 1])
  filters_dy = tf.reshape(flow_weights_dy, [1, 1, -1, 1, 1])
  lattice = tf.transpose(lattice, [0, 2, 1, 3, 4, 5])  # [b, x, y, dx, dy]
  lattice = tf.reshape(lattice, [FLAGS.batch_size*grid_size, grid_size,
                                 flow_grid_size, flow_grid_size, 1])  # [b*x, y, dx, dy]

  filtered = tf.nn.convolution(lattice, filters_y, padding='SAME')  # conv along y
  filtered = tf.reshape(filtered, [FLAGS.batch_size, grid_size, grid_size,
                                   flow_grid_size, flow_grid_size, 1])  # [b, x, y, dx, dy]
  filtered = tf.transpose(filtered, [0, 2, 1, 3, 4, 5])  # [b, y, x, dx, dy]
  filtered = tf.reshape(filtered, [FLAGS.batch_size*grid_size, grid_size,
                                   flow_grid_size, flow_grid_size, 1])  # [b*y, x, dx, dy]

  filtered = tf.nn.convolution(filtered, filters_x, padding='SAME')  # conv along x
  filtered = tf.nn.convolution(filtered, filters_dx, padding='SAME')  #conv along dx
  filtered = tf.nn.convolution(filtered, filters_dy, padding='SAME')  # conv along dy

  filtered = tf.reshape(filtered, [FLAGS.batch_size, -1])
  sliced_batch_idx = tf.expand_dims(tf.range(FLAGS.batch_size, dtype=tf.int64), 1)
  sliced_batch_idx = tf.reshape(tf.tile(sliced_batch_idx, [1, frame_shape[0]*frame_shape[1]]), [-1])
  slice_index = tf.stack((sliced_batch_idx, tf.reshape(slice_index, [-1])), axis=1)
  # print slice_index.get_shape()
  sliced = tf.gather_nd(filtered, slice_index)
  sliced = tf.reshape(sliced, [FLAGS.batch_size, frame_shape[0], frame_shape[1]])
  sliced = tf.nn.relu(sliced)
  sliced = tf.reshape(sliced, [FLAGS.batch_size, -1])
  labels = tf.reshape(labels, [FLAGS.batch_size, -1])
  objectness = tf.reshape(objectness, [FLAGS.batch_size, -1])
  labels = tf.to_float(labels)

  # Sigmoid cross entropy loss
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=1.-0.5*labels, logits=sliced)*tf.to_float(objectness>0.001)

  loss = tf.reduce_mean(loss)
  slim.losses.add_loss(loss)

  reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  slim.losses.add_loss(reg_loss)

  global_step = slim.get_or_create_global_step()
  learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                             global_step,
                                             FLAGS.steps_per_decay,
                                             FLAGS.learning_rate_decay,
                                             staircase=True)
  with tf.device("cpu:0"):
    slim.summaries.add_scalar_summaries(slim.losses.get_losses(), 'losses')
    slim.summaries.add_scalar_summary(
      slim.losses.get_total_loss(), 'Total Loss', 'losses')
    slim.summaries.add_scalar_summary(learning_rate, 'Learning Rate', 'training')
    slim.summaries.add_image_summary(tf.reshape(labels, [FLAGS.batch_size, 480, 854, 1]), 'label', 'Input')
    # vis = tf.maximum(0., sliced)
    vis = tf.sigmoid(sliced)
    slim.summaries.add_image_summary(tf.reshape(vis, [FLAGS.batch_size, 480, 854, 1]),
                                     'prediction', 'Output')
    slim.summaries.add_image_summary(tf.reshape(objectness,
                                                [FLAGS.batch_size, frame_shape[0], frame_shape[1], 1]),
                                     'objectness', 'Input')
    binarized = tf.to_float(objectness > sliced)
    slim.summaries.add_image_summary(tf.reshape(binarized,
                                                [FLAGS.batch_size, frame_shape[0], frame_shape[1], 1]),
                                     'Threshold', 'Output')
    slim.summaries.add_image_summary(tf.expand_dims(tf.expand_dims(weights, 0), 3),
                                     'Filter', 'Weight')
    slim.summaries.add_image_summary(tf.expand_dims(tf.expand_dims(loc_weights, 0), 3),
                                     'Filter', 'Weight_loc')

  optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
  optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer)

  variables_to_train = slim.get_trainable_variables()
  train_tensor = slim.learning.create_train_op(
      slim.losses.get_total_loss(),
      optimizer=optimizer,
      clip_gradient_norm=5.0,
      update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS),
      variables_to_train=variables_to_train)


  logging.info('start trainig')
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True

  tf.contrib.training.train(
      train_tensor,
      logdir=FLAGS.checkpoint_dir,
      master='',
      save_checkpoint_secs=FLAGS.save_checkpoint_secs,
      save_summaries_steps=10,
      config=sess_config,
  )


if __name__ == '__main__':
  app.run()