import os
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from vos import utils
from data import davis_flow_dataset

flags.DEFINE_string('dataset', 'Davis', 'Dataset used for training')
flags.DEFINE_string('split', 'val', 'Dataset split')
flags.DEFINE_integer('num_classes', 2, 'Number of classes')
flags.DEFINE_string('data_dir', './FlowExamples', 'Dir to TFExamples.')

flags.DEFINE_float('sigma', 3., 'Initial sigma for the Gaussian filter')
flags.DEFINE_integer('kernel_size', 21, 'kernel size')
flags.DEFINE_integer('loc_kernel_size', 37, 'kernel size of loc dim')
flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay factor.')


flags.DEFINE_string('checkpoint', None, 'Continue training from previous checkpoint')
flags.DEFINE_string('output_dir', None, 'Output dir')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_bool('save_raw_results', False, 'Save raw results')
flags.DEFINE_integer('random_select_k_bg', 50000,
                     'Random select k pixels from low objectness region for data augmentation')


FLAGS = flags.FLAGS


def create_2d_gaussian_kernel(kernel_size, sigma):
  half_window = (kernel_size-1)//2
  h_feat = np.arange(-half_window, half_window+1)
  w_feat = np.arange(-half_window, half_window+1)
  hw_feat = np.array(np.meshgrid(h_feat, w_feat, indexing='ij'))
  kernel = np.sum((hw_feat/sigma)**2, axis=0)
  kernel = np.exp(-kernel)
  kernel = kernel/np.sum(kernel)
  return kernel


def create_1d_uniform_kernel(kernel_size):
  kernel = np.ones(kernel_size)
  kernel /= np.sum(kernel)
  return kernel


def create_1d_gaussian_kernel(kernel_size, sigma):
  half_window = (kernel_size - 1) // 2
  h_feat = np.arange(-half_window, half_window + 1)
  kernel = np.exp(-0.5*(h_feat/sigma)**2)
  kernel = kernel/np.sum(kernel)
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
  # Create output dir. Shapes of 4-D tensors and frames are hard-coded.
  if not FLAGS.output_dir is None:
    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    global_mask_dir = os.path.join(FLAGS.output_dir, 'UnaryOnly')
    if not os.path.exists(global_mask_dir):
      os.mkdir(global_mask_dir)
    global_vis_dir = os.path.join(FLAGS.output_dir, 'UnaryVis')
    if not os.path.exists(global_vis_dir):
      os.mkdir(global_vis_dir)
    global_bnn_dir = os.path.join(FLAGS.output_dir, 'BNNRaw')
    if not os.path.exists(global_bnn_dir):
      os.mkdir(global_bnn_dir)
  flow_grid_size = 40
  grid_size = 18
  frame_shape = (480, 854)

  with tf.Graph().as_default():
    dataset = davis_flow_dataset.get_split(FLAGS.split, FLAGS.data_dir)
    # Use the no augmentation provider.
    (labels, lattice, slice_index, obj, sequence_name,
     timestep) = davis_flow_dataset.provide_data(
      dataset, shuffle=False, num_epochs=None, batch_size=FLAGS.batch_size)

    # Some duplicated code from train code.
    flow_init = create_1d_gaussian_kernel(FLAGS.kernel_size, FLAGS.sigma)
    loc_init = create_1d_uniform_kernel(kernel_size=FLAGS.loc_kernel_size)
    flow_weights_dx = tf.contrib.framework.variable('weights/dx',
                                                    shape=flow_init.shape,
                                                    initializer=tf.constant_initializer(flow_init),
                                                    regularizer=l2_regularizer(FLAGS.weight_decay),
                                                    trainable=True)
    flow_weights_dy = tf.contrib.framework.variable('weights/dy',
                                                    shape=flow_init.shape,
                                                    initializer=tf.constant_initializer(flow_init),
                                                    regularizer=l2_regularizer(FLAGS.weight_decay),
                                                    trainable=True)
    weights_x = tf.contrib.framework.variable('weights/x',
                                              shape=loc_init.shape,
                                              initializer=tf.constant_initializer(loc_init),
                                              regularizer=l2_regularizer(FLAGS.weight_decay),
                                              trainable=True)
    weights_y = tf.contrib.framework.variable('weights/y',
                                              shape=loc_init.shape,
                                              initializer=tf.constant_initializer(loc_init),
                                              regularizer=l2_regularizer(FLAGS.weight_decay),
                                              trainable=True)

    lattice = tf.expand_dims(lattice, 5)

    filters_y = tf.reshape(weights_y, [-1, 1, 1, 1, 1])
    filters_x = tf.reshape(weights_x, [-1, 1, 1, 1, 1])
    filters_dx = tf.reshape(flow_weights_dx, [1, -1, 1, 1, 1])
    filters_dy = tf.reshape(flow_weights_dy, [1, 1, -1, 1, 1])
    lattice = tf.transpose(lattice, [0, 2, 1, 3, 4, 5])  # [b, x, y, dx, dy]
    lattice = tf.reshape(lattice, [FLAGS.batch_size * grid_size, grid_size,
                                   flow_grid_size, flow_grid_size, 1])  # [b*x, y, dx, dy]

    filtered = tf.nn.convolution(lattice, filters_y, padding='SAME')  # conv along y
    filtered = tf.reshape(filtered, [FLAGS.batch_size, grid_size, grid_size,
                                   flow_grid_size, flow_grid_size, 1])  # [b, x, y, dx, dy]
    filtered = tf.transpose(filtered, [0, 2, 1, 3, 4, 5])  # [b, y, x, dx, dy]
    filtered = tf.reshape(filtered, [FLAGS.batch_size * grid_size, grid_size,
                                   flow_grid_size, flow_grid_size, 1])  # [b*y, x, dx, dy]

    filtered = tf.nn.convolution(filtered, filters_x, padding='SAME')  # conv along x
    filtered = tf.nn.convolution(filtered, filters_dx, padding='SAME')  # conv along dx
    filtered = tf.nn.convolution(filtered, filters_dy, padding='SAME')  # conv along dy


    filtered = tf.reshape(filtered, [FLAGS.batch_size, -1])
    sliced_batch_idx = tf.expand_dims(tf.range(FLAGS.batch_size, dtype=tf.int64), 1)
    sliced_batch_idx = tf.reshape(tf.tile(sliced_batch_idx, [1, frame_shape[0] * frame_shape[1]]), [-1])
    slice_index = tf.stack((sliced_batch_idx, tf.reshape(slice_index, [-1])), axis=1)
    sliced = tf.gather_nd(filtered, slice_index)
    sliced = tf.reshape(sliced, [FLAGS.batch_size, frame_shape[0], frame_shape[1]])
    sliced = tf.nn.relu(sliced)

    # Scale the results according to the number of bg pixels for splatting during training
    # The network is almost linear (without the last relu layer), if 100k pixels have obj < 0.001,
    # and used for splatting, the overall scale will not match traning, where only 50k pixels are
    # splatted onto the lattice.
    sliced = sliced * FLAGS.random_select_k_bg / tf.reduce_sum(lattice)
    sliced = tf.reshape(sliced, [FLAGS.batch_size, frame_shape[0], frame_shape[1]])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    restorer = tf.train.Saver(tf.global_variables())
    restorer.restore(sess, FLAGS.checkpoint)

    tf.train.start_queue_runners(sess=sess)
    for i in range(davis_flow_dataset._SPLITS_TO_SIZES[FLAGS.split]):
      [sliced_out, name_out, frame, obj_out] = sess.run(
        [sliced, sequence_name, timestep, obj])
      sliced_out = (1 / (1 + np.exp(-sliced_out)) - 0.5) * 2

      if not FLAGS.output_dir is None:
        if FLAGS.save_raw_results:
          output_dir = os.path.join(FLAGS.output_dir, name_out[0])
          if not os.path.exists(output_dir):
            os.mkdir(output_dir)
          np.save(os.path.join(output_dir, '%05d.npy'%frame[0]), sliced_out)
        mask_dir = os.path.join(global_mask_dir, name_out[0])
        if not os.path.exists(mask_dir):
          os.mkdir(mask_dir)

        mask = np.squeeze(obj_out > sliced_out)
        cv2.imwrite(os.path.join(mask_dir, '%05d.png'%frame[0]), mask.astype(np.uint8)*255)
        vis_dir = os.path.join(global_vis_dir, name_out[0])
        if not os.path.exists(vis_dir):
          os.mkdir(vis_dir)
        cv2.imwrite(os.path.join(vis_dir, '%05d.jpg' % frame[0]),
                    utils.get_heatmap(sliced_out))
        bnn_dir = os.path.join(global_bnn_dir, name_out[0])
        if not os.path.exists(bnn_dir):
          os.mkdir(bnn_dir)
        raw_mask = np.squeeze(0.5 > sliced_out)
        cv2.imwrite(os.path.join(bnn_dir, '%05d.png' % frame[0]),
                    raw_mask.astype(np.uint8) * 255)


if __name__ == '__main__':
  app.run()
