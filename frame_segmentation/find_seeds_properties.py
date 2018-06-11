import os
import numpy as np
import glob
import cv2
import argparse
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
import scipy.io as sio

import cfg
import utils


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Extract seeds for VOS')
  parser.add_argument('--set', dest='set',
                      help='image set',
                      default='val', type=str)
  parser.add_argument('--output_dir', dest='output_dir',
                      help='the output dir',
                      default=None, type=str)
  parser.add_argument('--img_dir', dest='img_dir',
                      help='the directory of frame',
                      default=None, type=str)
  parser.add_argument('--obj_dir', dest='obj_dir', help='objectness directory',
                      default='/home/siyang/backup/VOS_data/objectness', type=str)
  parser.add_argument('--bg_dir', dest='bg_dir', help='static region prediction directory',
                      default=None, type=str)
  parser.add_argument('--flow_dir', dest='flow_dir', help='optical flow directory',
                      default=None, type=str)
  parser.add_argument('--embedding_dir', dest='embedding_dir',
                      help='embedding directory',
                      default=None,
                      type=str)
  parser.add_argument('--seed_window', dest='seed_window',
                      help='the window for local minima', default=9,
                      type=int)

  args = parser.parse_args()
  return args


def vos_frame(objectness_map, bg_prob, features,
              flow, seed_window=9, prev_label_map=None):
  """Extract seeds and features for graph cut step.
  Args:
    objectness_map: the objectness score for each pixel.
    bg_prob: the background score from bilateral NN.
    features: the feature map (embedding map).
    flow: the optical flow map
    seed_window: the local minimum window used to extract seeds.
    prev_label_map: the label map of the previous frame.
  Returns:

  """
  features = np.squeeze(features)
  height = features.shape[0]
  width = features.shape[1]
  objectness_map = cv2.resize(objectness_map, (width, height), interpolation=cv2.INTER_LINEAR)
  seeds = utils.kpp_seeds(features, objectness_map, window=seed_window)

  # Oversegment for seed adjacency matrix
  label_map = distance_transform_edt(np.logical_not(seeds),
                                     return_distances=False,
                                     return_indices=True)



  sp_masks = []
  seeds_sparse = [(i, j) for i, j in zip(*seeds.nonzero())]
  adjacent_matrix = np.zeros((len(seeds_sparse), len(seeds_sparse)), dtype=np.bool)
  for i in range(0, np.max(label_map.astype(np.int32))):
    region = label_map == i
    sp_masks.append(region)

    # find neighbors
    region_edge = binary_dilation(region) & np.logical_not(region)
    neighbors = np.unique(utils.select_seed_embedding(region_edge, label_map))
    for n in neighbors:
      adjacent_matrix[i, n] = True

  flow_height = flow.shape[0]
  flow_width = flow.shape[1]
  flow_padded = np.concatenate([flow, np.zeros((flow_height, flow_width, 1))], axis=2)
  flow_padded = cv2.resize(flow_padded, (width, height),
                           interpolation=cv2.INTER_LINEAR)
  flow = flow_padded[:, :, :2]

  bg_prob = cv2.resize(bg_prob, (width, height),
                      interpolation=cv2.INTER_LINEAR)

  if not prev_label_map is None:
    adj_with_last = np.zeros((np.max(label_map.astype(np.int32)) + 1,
                              np.max(prev_label_map.astype(np.int32)) + 1), dtype=np.bool)
    for i in range(0, np.max(label_map.astype(np.int32))):
      region = label_map == i

      # find neighbors
      neighbors = np.unique(utils.select_seed_embedding(region, prev_label_map))
      for n in neighbors:
        adj_with_last[i, n] = True
  else:
    adj_with_last = 0

  seeds_obj = np.array(utils.select_seed_embedding(seeds, objectness_map))
  seeds_bg = np.array(utils.select_seed_embedding(seeds, bg_prob))

  seeds_embed = np.array(utils.select_seed_embedding(seeds, features))
  seeds_flow = np.array(utils.select_seed_embedding(seeds, flow))
  seeds_loc = np.array(seeds_sparse)
  seed_feature_map = {'seeds_obj': seeds_obj,
                     'seeds_bg': seeds_bg,
                     'seeds_loc': seeds_loc,
                     'seeds_embed': seeds_embed,
                     'seeds_flow': seeds_flow,
                     'adjacency': adjacent_matrix,
                     'label_map': label_map,
                     'adj_with_last': adj_with_last,
                     }

  return seed_feature_map


def process_set(sequence_name_list, output_dir,
                img_dir, bg_dir, obj_dir, embedding_dir,
                flow_dir, seed_window=9):
  for s in sequence_name_list:


    length = len(glob.glob('%s/%s/*' %(img_dir, s)))
    print length

    frame = 0
    prev_label_map = None
    # The last frame is not processed because optical flow is not available.
    # This is allowed under DAVIS evaluation protocol.
    while frame < length - 1:
      if frame % 10 == 0:
        print '%d/%d' % (frame, length)
      # Read data
      objectness_map = np.load(os.path.join(obj_dir, s, '%05d.npy' % frame))
      bg_prob = np.load(os.path.join(bg_dir, '%s/%05d.npy' % (s, frame)))
      features = np.load(os.path.join(embedding_dir, '%s_%05d.npy' % (s, frame)))
      flow = np.load(os.path.join(flow_dir, '%s_%05d.npy' % (s, frame)))

      seed_feature_map = vos_frame(objectness_map, bg_prob, features,
                                   flow, seed_window, prev_label_map)
      prev_label_map = seed_feature_map['label_map']

      # Save output
      if output_dir:
        sequence_output = os.path.join(output_dir, s)
        if not os.path.exists(sequence_output):
          os.mkdir(sequence_output)
        sio.savemat(os.path.join(sequence_output, '%05d.mat' % frame),
                    mdict=seed_feature_map)
      frame += 1


def main(args):
  if not os.path.exists(args.output_dir):
    print args.output_dir
    os.mkdir(args.output_dir)
  if args.set == 'train':
    set_list = cfg.train_list
  elif args.set == 'val':
    set_list = cfg.val_list

  elif args.set == 'trainval':
    set_list = cfg.val_list + cfg.train_list
  else:
    raise ValueError("Video set not found.")
  process_set(set_list, args.output_dir, args.img_dir,
              args.bg_dir, args.obj_dir, args.embedding_dir,
              args.flow_dir, args.seed_window)


if __name__ == '__main__':
  args = parse_args()
  main(args)
