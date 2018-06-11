import os
import numpy as np
import cv2
import argparse
import glob
import scipy.io as sio
from sklearn.metrics.pairwise import euclidean_distances

import cfg
import utils


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Unsupervised VOS')
  parser.add_argument('--set', dest='set',
                      help='image set',
                      default='val', type=str)
  parser.add_argument('--output_dir', dest='output_dir',
                      help='the output dir',
                      default=None, type=str)
  parser.add_argument('--img_dir', dest='img_dir',
                      help='the directory of frame',
                      default=None, type=str)
  parser.add_argument('--anno_dir', dest='anno_dir',
                      help='the directory of annotation',
                      default=None, type=str)
  parser.add_argument('--embedding_dir', dest='embedding_dir',
                      help='embedding directory',
                      default=None, type=str)
  parser.add_argument('--obj_dir', dest='obj_dir',
                      help='objectness directory', default=None, type=str)
  parser.add_argument('--gc_dir', dest='gc_dir',
                      help='graph cut results directory', default=None, type=str)
  parser.add_argument('--seeds_dir', dest='seeds_dir',
                      help='seeds directory', default=None, type=str)
  parser.add_argument('--crf', dest='crf',
                      help='appy CRF or not', default=0, type=int)
  parser.add_argument('--sigma', dest='sigma',help='Used to compute soft FG prob',
                      default=1., type=float)
  args = parser.parse_args()
  return args


def slicing(features, seeds_features, seeds_label, label_map, adjacency,
            sigma=1., resize_shape=(480, 854)):
  """
  Propagate the labels of seeds to pixels.
  Similiar with the slicing step in bilateral filtering.
  Args:
    features: dense feature map [height, width, f_dim]
    seeds_features: [num_seeds_cur_prev_following_frame, f_dim]
    seeds_label: [num_seeds_cur_prev_following_frame]
    label_map: [height, width]
    adjacency: [num_seeds_current, num_seeds_cur_prev_following_frame]
    sigma: Used to compute the soft prob of FG
    resize_shape: Resize to original resolution
  Returns:
    prob: soft FG/BG probability map
    dist_vis: visualized distance map
  """
  label_map_flatten = np.reshape(label_map, [-1])
  num_seeds = np.max(label_map)+1
  # Label_map_one_hot [num_pixels, num_seeds_current]
  label_map_one_hot = np.zeros((label_map_flatten.shape[0], num_seeds), dtype=np.int16)
  label_map_one_hot[np.arange(label_map_flatten.shape[0]), label_map_flatten] = 1
  # weight_idx: [num_pixels, num_seeds_cur_prev_following_frame]
  # Only neighbouring seeds have weights > 0
  weight_idx = np.matmul(label_map_one_hot, adjacency)
  feature_dim = features.shape[2]

  # This implementation is not very efficient
  # It computes pairwise distance between all pixels and all seeds (from 3 frames)
  # dist: [num_pixels, num_seeds_cur_prev_following_frame]
  dist = euclidean_distances(np.reshape(features, [-1, feature_dim]), seeds_features)
  weight = np.exp(-dist*dist/sigma/sigma)
  weight *= weight_idx
  fg_votes = np.max(weight*np.expand_dims(seeds_label==1, 0), axis=1)
  bg_votes = np.max(weight*np.expand_dims(seeds_label==0, 0), axis=1)
  height = features.shape[0]
  width = features.shape[1]
  fg_votes = fg_votes.reshape((height, width))+1e-8
  bg_votes = bg_votes.reshape((height, width))+1e-8
  fg_votes = cv2.resize(fg_votes, (resize_shape[1], resize_shape[0]),
                        interpolation=cv2.INTER_LINEAR)
  bg_votes = cv2.resize(bg_votes, (resize_shape[1], resize_shape[0]),
                        interpolation=cv2.INTER_LINEAR)

  prob = np.stack([bg_votes, fg_votes], axis=2)
  dist_vis = utils.get_heatmap(np.concatenate([fg_votes, bg_votes], axis=0))
  prob = prob/np.sum(prob, axis=2, keepdims=True)

  return prob, dist_vis


def vos_frame(name, frame, img_dir, embedding_dir, gc_dir, seeds_dir, end=False):
  # Read data
  features = np.load('%s/%s/%05d.npy' % (embedding_dir, name, frame))
  seg_labels = sio.loadmat(os.path.join(gc_dir, name, '%05d.mat' % frame))
  seeds_prop = sio.loadmat(os.path.join(seeds_dir, name, '%05d.mat' % frame))

  features = np.squeeze(features)
  # Get the labeled seeds from graph cut
  fg_seeds_idx = np.squeeze(seg_labels['seg_labels_this_frame'].astype(np.int32))
  label_map = seeds_prop['label_map']  # The region map (index of the seed for each pixel it belongs to
  adjacency = seeds_prop['adjacency']  # Region adjacency
  adjacency = np.logical_or(adjacency, np.eye(np.max(label_map)+1))
  seeds_features = seeds_prop['seeds_embed']


  # Consider the previous, current and following frames
  last_frame = frame - 1
  if last_frame >= 0:
    l_seg_labels = sio.loadmat(os.path.join(args.gc_dir, name, '%05d.mat' % last_frame))
    l_seeds_prop = sio.loadmat(os.path.join(args.seeds_dir, name, '%05d.mat' % last_frame))
    last_seeds_features = l_seeds_prop['seeds_embed']
    seeds_features = np.concatenate([seeds_features, last_seeds_features], axis=0)
    fg_seeds_idx = np.concatenate([fg_seeds_idx,
                                   np.squeeze(l_seg_labels['seg_labels_this_frame'].astype(np.int32))],
                                  axis=0)
    adjacency = np.concatenate([adjacency, seeds_prop['adj_with_last']], axis=1)
  if not end:
    next_frame = frame + 1
    n_seg_labels = sio.loadmat(os.path.join(args.gc_dir, name, '%05d.mat' % next_frame))
    n_seeds_prop = sio.loadmat(os.path.join(args.seeds_dir, name, '%05d.mat' % next_frame))
    n_seeds_features = n_seeds_prop['seeds_embed']
    seeds_features = np.concatenate([seeds_features, n_seeds_features], axis=0)
    fg_seeds_idx = np.concatenate([fg_seeds_idx,
                                   np.squeeze(n_seg_labels['seg_labels_this_frame'].astype(np.int32))],
                                  axis=0)
    adjacency = np.concatenate([adjacency, np.transpose(n_seeds_prop['adj_with_last'])], axis=1)

  # Label propagate
  prob, dist_vis = slicing(features, seeds_features, fg_seeds_idx, label_map, adjacency, sigma=args.sigma)

  if args.crf > 0:
    image = cv2.imread('%s/%s/%05d.png' % (img_dir, name, frame))
    seg = utils.apply_crf(prob, image, shape=image.shape[0:2])
  else:
    seg = prob[:, :, 1] > prob[:, :, 0]

  return seg


def process_set(sequence_name_list, output_dir, img_dir,
                anno_dir, embedding_dir, gc_dir, seeds_dir):
  result_list = []
  result_log = []
  for s in sequence_name_list:
    if output_dir:
      sequence_output = os.path.join(output_dir, s)
      if not os.path.exists(sequence_output):
        os.mkdir(sequence_output)
    else:
      sequence_output = None

    frame = 0
    length = len(glob.glob('%s/%s/*.jpg' % (img_dir, s)))
    seg_list = []
    gt_list = []

    # The last frame is not processed, allowed according to DAVIS evaluation.
    while frame < length - 1:
      if frame % 10 == 0:
        print '%d/%d' % (frame, length)
      gt = cv2.imread('%s/%s/%05d.png' % (anno_dir, s, frame),
                      flags=cv2.IMREAD_GRAYSCALE)
      gt = (gt > 0).astype(np.uint8)
      gt_list.append(gt)

      seg = vos_frame(s, frame, img_dir, embedding_dir,
                      gc_dir, seeds_dir, frame == length - 2)

      if sequence_output:
        cv2.imwrite('%s/seg_%05d.png' % (sequence_output, frame),
                      (seg * 255).astype(np.uint8))

      seg_list.append(seg)
      frame += 1

    # Evaluation
    iou = utils.compute_iou(seg_list, gt_list)
    mean_iou = np.average(iou)
    result_log.append('%s\t%.5f\n'%(s, mean_iou))
    print s, np.average(iou)
    result_list.append(np.average(iou))
  result_log.append('avg\t%.5f\b'%(np.sum(np.array(result_list))/len(result_list)))
  print 'avg\t', np.sum(np.array(result_list))/len(result_list)
  return result_log


def main(args):
  if not os.path.exists(args.output_dir):
    print args.output_dir
    os.mkdir(args.output_dir)
  if args.set == 'train':
    sequence_list = cfg.train_list
  elif args.set == 'val':
    sequence_list = cfg.val_list
  else:
    raise ValueError('Video set not found.')

  log = process_set(sequence_list, args.output_dir, args.img_dir,
                    args.anno_dir, args.embedding_dir, args.gc_dir, args.seeds_dir)
  with open(os.path.join(args.output_dir, '%s_log.txt' % args.set), 'w') as f:
    f.writelines(log)


if __name__ == '__main__':
  args = parse_args()
  main(args)

