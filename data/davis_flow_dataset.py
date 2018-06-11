"""Provide optical flow data (DAVIS 2016 dataset)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import tensorflow as tf
from tensorflow.contrib.slim import dataset
from tensorflow.contrib.slim import dataset_data_provider
from tensorflow.contrib.slim import tfexample_decoder

# These are total number of frames, not the number of sequences.
_SPLITS_TO_SIZES = {'train': 2049, 'trainval': 3405, 'val': 1356}
_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'height': 'Height of image',
    'width': 'Width of image',
    'sequence_name': 'Name of sequence',
    'timestep': 'The timestep of the current frame',
    'object_labels': 'Ground truth of BG-FG segmentation',
    'flow_lattice': 'splatted flow lattice',
    'lattice_height': 'height of lattice',
    'lattice_width': 'width of lattice',
    "slice_index": 'slice_index',
    "objectness": 'semantic_map',
}


def get_split(split_name, dataset_dir):
  """Get the dataset object for DAVIS 2016.

  Note that the existence of data files is NOT checked here.

  Args:
    split_name: 'train', 'trainval' or 'val'.
    dataset_dir: The directory of the dataset sources.
  Returns:
    A dataset object.
  Raises:
    ValueError: if split_name is not recognized.
  """

  file_pattern = os.path.join(dataset_dir, '%s*' % split_name)

  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s not found.' % split_name)

  # Parse tfexamples.
  # "flow/slice_index" specifies the flattened index in the
  #  4-D bilateral tensor for each pixel, according to its (dx, dy, x, y)
  keys_to_features = {
      'flow/height':
          tf.FixedLenFeature((), tf.int64, default_value=0),
      'flow/width':
          tf.FixedLenFeature((), tf.int64, default_value=0),
      'sequence/timestep':
          tf.FixedLenFeature((), tf.int64, default_value=0),
      'sequence/name':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/segmentation/object/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/segmentation/object/format':
          tf.FixedLenFeature((), tf.string),
      "flow_lattice/height":
          tf.FixedLenFeature((), tf.int64, default_value=0),
      "flow_lattice/width":
        tf.FixedLenFeature((), tf.int64, default_value=0),
      "flow_lattice/values":
          tf.VarLenFeature(tf.float32),
      "flow/slice_index":  # See comments above.
        tf.VarLenFeature(tf.int64),
      "prediction/objectness": tf.VarLenFeature(tf.float32),

  }

  # Handle each feature.
  items_to_handlers = {
      'height':
          tfexample_decoder.Tensor('flow/height'),
      'width':
          tfexample_decoder.Tensor('flow/width'),
      'flow_lattice':
        tfexample_decoder.Tensor('flow_lattice/values', default_value=0.),
      'lattice_height':
        tfexample_decoder.Tensor('flow_lattice/height'),
      'lattice_width':
        tfexample_decoder.Tensor('flow_lattice/width'),
      'sequence_name':
          tfexample_decoder.Tensor('sequence/name'),
      'timestep':
          tfexample_decoder.Tensor('sequence/timestep'),
      'object_labels':
          tfexample_decoder.Image(
              'image/segmentation/object/encoded',
              'image/segmentation/object/format',
              channels=1),
      'slice_index':
          tfexample_decoder.Tensor('flow/slice_index'),
      'objectness': tfexample_decoder.Tensor('prediction/objectness'),
  }

  decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,
                                               items_to_handlers)
  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES)



def provide_data(flow_dataset, shape=(480, 854), shuffle=False, num_epochs=None, batch_size=64):
  """Provide batched 4-D bilateral tensors for training without data augmentation. """

  davis_data_provider = dataset_data_provider.DatasetDataProvider(
      flow_dataset, shuffle=shuffle, num_epochs=num_epochs)
  (height, width, sequence_name,
   timestep, object_labels, lattice_flattened,
   lattice_height, lattice_width, slice_index, objectness) = davis_data_provider.get(
       ['height', 'width', 'sequence_name', 'timestep', 'object_labels',
        'flow_lattice', 'lattice_height', 'lattice_width', 'slice_index', 'objectness'])
  lattice = tf.reshape(lattice_flattened, (18, 18, 40, 40))
  object_labels = tf.to_int32(tf.squeeze(object_labels, [2]))
  object_labels.set_shape(shape)
  objectness = tf.reshape(objectness, shape)
  slice_index.set_shape((480*854))
  batch_data = []
  batch_data.append([lattice, object_labels, slice_index, objectness, sequence_name, timestep])
  lattice, object_labels, slice_index, objectness, sequence_name, timestep = tf.train.batch_join(
    batch_data,
    batch_size=batch_size,
    capacity= 2 * batch_size)

  return object_labels, lattice, slice_index, objectness, sequence_name, timestep


# The frame size and lattice size is hard-coded,
# as I don't know how to set it to flexible shapes.
def provide_data_augmented(flow_dataset, shape=(480, 854), shuffle=False,
                           num_epochs=None, batch_size=64, random=True, random_select_k=50000):
  """Provide batched 4-D bilateral tensors for training with data augmentation. """

  davis_data_provider = dataset_data_provider.DatasetDataProvider(
    flow_dataset, shuffle=shuffle, num_epochs=num_epochs)
  (height, width, sequence_name,
   timestep, object_labels, lattice_height, lattice_width,
   slice_index, objectness) = davis_data_provider.get(
    ['height', 'width', 'sequence_name', 'timestep', 'object_labels',
     'lattice_height', 'lattice_width', 'slice_index', 'objectness'])
  object_labels = tf.to_int32(tf.squeeze(object_labels, [2]))
  object_labels.set_shape(shape)
  objectness = tf.reshape(objectness, shape)
  slice_index.set_shape((480*854))

  lattice_shape = (18, 18, 40, 40)
  lattice = splat(slice_index, objectness, lattice_shape, random, random_select_k)  # A flattened 4-D tensor
  lattice = tf.reshape(lattice, lattice_shape)  # reshape to 4-D

  batch_data = []
  batch_data.append([lattice, object_labels, slice_index, objectness, sequence_name, timestep])
  lattice, object_labels, slice_index, objectness, sequence_name, timestep = tf.train.batch_join(
    batch_data,
    batch_size=batch_size,
    capacity=2 * batch_size)

  return object_labels, lattice, slice_index, objectness, sequence_name, timestep


def splat(slice_index, objectness, lattice_shape, random=True, random_select_k=50000, img_shape=(480, 854)):
  loc_grid = lattice_shape[0]
  flow_grid = lattice_shape[2]

  obj_flattened = tf.reshape(objectness, [-1])

  if random:
    # Data augmentation: it turns out that background propagation is location-ignorant,
    # i.e., the similarity of optical flow is more important. Thus the random selection
    # for augmentation is different from the normal way. See comments below.
    # It's probably OK to use motion network instead of bilateral network (motion + location).

    # Random select "random_select_k" pixels from low objectness
    # region (objectness < 0.001).
    # Data augmentation is done by random picking one pixel
    # from low obj region, and then finding the nearest "random_select_k" pixels
    # to this pixel within the low obj region
    obj_mask = obj_flattened < 0.001
    hw_idx = tf.range(img_shape[0]*img_shape[1])
    h = tf.to_float(tf.floor_div(hw_idx, img_shape[1]))
    w = tf.to_float(hw_idx) - h*float(img_shape[1])

    ptw = tf.random_uniform(shape=(),maxval=img_shape[1])
    pth = tf.random_uniform(shape=(), maxval=img_shape[0])

    d = tf.abs(ptw - w)+ tf.abs(pth - h)

    slice_bg_idx = tf.boolean_mask(slice_index, obj_mask)

    slice_bg_d = tf.boolean_mask(d, obj_mask)

    # select the nearest k point according to d
    _, nearest_k = tf.nn.top_k(-slice_bg_d, random_select_k)
    selected_idx = tf.gather(slice_bg_idx, nearest_k)
  else:
    _, top_k_idx = tf.nn.top_k(-obj_flattened, random_select_k)
    selected_idx = tf.gather(slice_index, top_k_idx)
  splatted = tf.bincount(tf.to_int32(selected_idx),
                         minlength=flow_grid ** 2 * loc_grid ** 2)
  splatted = tf.to_float(splatted)
  return splatted
