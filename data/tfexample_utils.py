"""Utils for TF example construction."""

import numpy as np
import tensorflow as tf


def create_tfexample(feature_maps):
  """Create tfexample given corresponding features.

  Args:
    feature_maps: a dictionary {feature_key_1: value_1, feature_key_2, value_2, ...}.
                  Value types can be scalars, numpys, lists, tf.train.Feature.
  Returns:
    example: a TF example.
  """
  tfexample_features = {}
  for key in feature_maps:
    tfexample_features[key] = convert_to_tf_feature(feature_maps[key])
  example = tf.train.Example(features=tf.train.Features(feature=tfexample_features))
  return example


def convert_to_tf_feature(value):
  """Convert the feature content to tf.train.Feature.

  Args:
    value: The feature values.
           Value types can be scalars, numpys, lists, tf.train.Feature.
  Returns:
    feature: tf.Train.Feature object.
  """
  if isinstance(value, tf.train.Feature):
    return value

  if isinstance(value, list):
    if len(value) == 0:
      raise TypeError("Cannot recognize data type")
    else:
      content_type = type(value[0])
      value_list = value
  elif isinstance(value, np.ndarray):
    content_type = value.dtype.type
    value_list = (np.ravel(value)).tolist()
  else:  # scalar
    content_type = type(value)
    value_list = [value]

  if np.issubdtype(content_type, np.float):
      feature = tf.train.Feature(float_list=tf.train.FloatList(value=value_list))
  elif np.issubdtype(content_type, np.integer):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))
  elif content_type is str:
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))
  else:
    raise TypeError("Cannot recognize data type")

  return feature