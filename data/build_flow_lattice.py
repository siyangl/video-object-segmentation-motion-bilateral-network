"""Build training TF examples"""

import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf

import tfexample_utils
from frame_segmentation import cfg
from frame_segmentation import utils


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Build flow-location dataset')
  parser.add_argument('--set', dest='set',
                      help='image set',
                      default='train', type=str)
  parser.add_argument('--output_dir', dest='output_dir',
                      help='the output dir',
                      default=None, type=str)
  parser.add_argument('--obj_dir', dest='obj_dir',
                      help='objectness directory',
                      default='/home/siyang/backup/VOS_data/objectness', type=str)
  parser.add_argument('--flow_file_pattern', dest='flow_file_pattern',
                      help='flow file pattern',
                      default='/home/siyang/backup/VOS_data/flow/%s_%05d.npy',
                      type=str)
  parser.add_argument('--anno_dir', dest='anno_dir', help='dir of GT mask',
                      default='DAVIS/Annotations/480p', type=str)
  parser.add_argument('--flow_var_thres', dest='flow_var_thres',
                      help='thres to remove probably badly computated flow',
                      default=0.15, type=float)
  parser.add_argument('--obj_thres', dest='obj_thres',
                      help='low objectness region threshold',
                      default=0.001, type=float)
  args = parser.parse_args()
  return args


def process_flow(name, frame):
  semantic_map = np.load(os.path.join(args.obj_dir, name, '%05d.npy'%frame))
  flow = np.load(args.flow_file_pattern % (name, frame))

  flow_edge = utils.compute_feature_edge(flow, use_max=False)
  flow_edge = 1 - np.exp(-flow_edge ** 2)
  flow_var = utils.compute_spatial_variance(flow_edge)
  if flow_var > args.flow_var_thres:
    print name, frame, flow_var
    return
  height = flow.shape[0]
  width = flow.shape[1]
  semantic_map = cv2.resize(semantic_map, (width, height),
                            interpolation=cv2.INTER_LINEAR)

  mask = semantic_map < args.obj_thres

  # Hard code bilateral grid size
  grid_size = 18
  flow_grid_size = 40

  # Create grid
  x = np.linspace(0, width, grid_size)
  y = np.linspace(0, height, grid_size)
  dy = np.linspace(np.min(flow[:, :, 0]), np.max(flow[:, :, 0]), flow_grid_size)
  dx = np.linspace(np.min(flow[:, :, 1]), np.max(flow[:, :, 1]), flow_grid_size)
  coord_feature = np.array(np.meshgrid(y, x, dy, dx, indexing='ij'))

  h_feat = np.arange(height)
  w_feat = np.arange(width)
  hw_feat = np.array(np.meshgrid(h_feat, w_feat, indexing='ij'))
  hw_feat = np.transpose(hw_feat, (1, 2, 0))

  feat = np.concatenate((hw_feat, flow), axis=2)

  # Step size of on each dim
  cell_size = coord_feature[:, 1, 1, 1, 1] - coord_feature[:, 0, 0, 0, 0]
  cell_min = np.array((0, 0, np.min(flow[:, :, 0]), np.min(flow[:, :, 1])))
  cell_size = np.expand_dims(cell_size, axis=0)
  cell_min = np.expand_dims(cell_min, axis=0)
  max_idx = np.expand_dims(np.array((grid_size, grid_size, flow_grid_size, flow_grid_size))-2,
                            axis=0)
  feat = np.reshape(feat, [-1, 4])
  feat_coord = (feat - cell_min) / cell_size
  feat_coord_lb = np.minimum(np.floor(feat_coord), max_idx)
  offset = (feat_coord-feat_coord_lb)

  # Find the nearest neighbor on the grid for each pixel (flattened grid index)
  offset = (offset > 0.5).astype(np.int)
  feat_coord = feat_coord_lb.astype(np.int) + offset
  feat_coord_flattend = flow_grid_size*flow_grid_size*grid_size*feat_coord[:, 0] + \
                        flow_grid_size * flow_grid_size * feat_coord[:, 1] + \
                        flow_grid_size * feat_coord[:, 2] + \
                        feat_coord[:,3]

  splatted = np.bincount(feat_coord_flattend, np.reshape(mask.astype(np.float), [-1]),
                         minlength=flow_grid_size**2 * grid_size ** 2)
  splatted = np.reshape(splatted, [grid_size, grid_size, flow_grid_size, flow_grid_size])

  annotation_file = os.path.join(args.anno_dir, name, '%05d.png' % frame)
  anno = cv2.imread(annotation_file, flags=cv2.IMREAD_GRAYSCALE)
  anno = anno // 255
  assert anno.shape == semantic_map.shape
  anno_encoded = cv2.imencode(".png", anno.astype(np.uint8))[1].tostring()
  features = {
    "sequence/name": name,
    "sequence/timestep": frame,
    "flow_lattice/height": flow_grid_size,
    "flow_lattice/width": flow_grid_size,
    "flow_lattice/values": splatted,
    "flow/channels": 2,
    "flow/height": height,
    "flow/width": width,
    "image/segmentation/object/encoded": anno_encoded,
    "image/segmentation/object/format": "png",
    "flow/slice_index": feat_coord_flattend,
    "prediction/objectness": semantic_map,
  }
  example = tfexample_utils.create_tfexample(features)

  return example


def process_set(sequence_name_list, output_path):
  writer = tf.python_io.TFRecordWriter(output_path)
  for s in sequence_name_list:
    invalid_frames = 0
    frame = 0
    length = len(glob.glob('DAVIS/Annotations/480p/%s/*' % s))
    print length
    while frame < length - 1:
      if frame % 10 == 0:
        print '%d/%d' % (frame, length)
      example = process_flow(s, frame)
      if not example is None:
        writer.write(example.SerializeToString())
      else:
        invalid_frames += 1
        print "not writing frame %d of %s" % (frame, s)
      frame += 1
    print('not using %d frames' % invalid_frames)
    print("Finish '%s' to TFRecords." % s)

  print("Finish writing data to TFRecords.")


if __name__ == '__main__':
  args = parse_args()
  if not os.path.exists(args.output_dir):
    print args.output_dir
    os.mkdir(args.output_dir)
  log = []
  output_path = os.path.join(args.output_dir, args.set)

  if args.set == 'train':
    log = process_set(cfg.train_list, output_path)
  elif args.set == 'val':
    log = process_set(cfg.val_list, output_path)
  elif args.set == 'trainval':
    process_set(cfg.val_list + cfg.train_list, output_path)
