"""til functions."""

import numpy as np
import copy
import cv2

from sklearn.metrics.pairwise import euclidean_distances
import scipy.ndimage.filters as filters


import pydensecrf.densecrf as dcrf


def kpp_seeds(features, objectness, k=100, window=9):
  height = features.shape[0]
  width = features.shape[1]
  seeds = np.zeros((height, width), dtype=np.bool)
  edge = compute_feature_edge(features)
  seeds_candidates = (filters.minimum_filter(edge, window) == edge)

  seeds_candidates_idx = [(i, j) for i, j in zip(*seeds_candidates.nonzero())]
  seed_obj = select_seed_embedding(seeds_candidates, objectness)
  start = np.argmax(seed_obj)

  seeds[seeds_candidates_idx[start][0], seeds_candidates_idx[start][1]] = True
  feature_flatten = select_seed_embedding(seeds_candidates, features)
  count = 1
  while count < k:
    seeds_features = select_seed_embedding(seeds, features)
    dist_pair = euclidean_distances(seeds_features, feature_flatten)
    next_seed = np.argmax(np.min(dist_pair, axis=0))
    seeds[seeds_candidates_idx[next_seed][0], seeds_candidates_idx[next_seed][1]] = True
    count += 1
  return seeds


def compute_spatial_variance(features):
  """Compute spatial variance for a feature map (h, w, channel)."""

  height = features.shape[0]
  width = features.shape[1]
  h_feat = np.arange(height, dtype=np.float)/(height-1)
  w_feat = np.arange(width, dtype=np.float)/(width-1)
  hw_feat = np.array(np.meshgrid(h_feat, w_feat, indexing='ij'))
  hw_feat = np.transpose(hw_feat, (1, 2, 0))

  norm = np.sum(features)
  y_mean = np.sum(hw_feat[:, :, 0]*features)/norm
  x_mean = np.sum(hw_feat[:, :, 1]*features)/norm
  centroid = np.array([y_mean, x_mean]).reshape((1, 1, 2))
  var = np.sum(np.sum((hw_feat - centroid)**2, axis=2)*features)/norm
  return var


def compute_feature_edge(features, use_max=False):
  """Feature edge by finding the Euclidean distance to its 4 neighbors."""

  h1 = np.zeros(features.shape[0:2])
  h2 = np.zeros(features.shape[0:2])
  hd = np.linalg.norm(features[1:] - features[:-1], axis=2)
  h1[1:, :] = hd
  h2[:-1, :] = hd
  v1 = np.zeros(features.shape[0:2])
  v2 = np.zeros(features.shape[0:2])
  vd = np.linalg.norm(features[:, 1:, :] - features[:, :-1, :], axis=2)
  v1[:, 1:] = vd
  v2[:, :-1] = vd
  if use_max:
    edge = np.max(np.stack((h1, h2, v1, v2), axis=2), axis=2)
  else:
    edge = np.average(np.stack((h1, h2, v1, v2), axis=2), axis=2)
  return edge


def get_heatmap(values, normalize=False):
  """Visualize a one channel feature map (JET heatmap)."""
  values_copy = copy.deepcopy(values)
  values_copy = values_copy.astype(np.float32)

  if np.min(values_copy) < 0:
    values_copy -= np.min(values)
  # values_copy = np.minimum(values_copy, 1)
  if np.max(values_copy) > 1 or normalize:
    values_copy /= np.max(values_copy)
  vis = cv2.applyColorMap((values_copy * 255).astype(np.uint8),
                          cv2.COLORMAP_JET)
  return vis


def select_seed_embedding(seeds, embeddings):
  """Select the embedding at certain locations."""
  if embeddings.ndim > 2:
    embeddings = np.reshape(embeddings, [-1, embeddings.shape[-1]])
  else:
    embeddings = np.reshape(embeddings, [-1])
  seeds = np.reshape(seeds, [-1])
  if embeddings.ndim > 2:
    selected_embedding = embeddings[seeds, :]
  else:
    selected_embedding = embeddings[seeds]
  return selected_embedding


def compute_iou(pred_list, gt_list):
  """IoU between prediction and ground truth.
  Both are interpreted as a list of 2-D label maps."""

  pred = np.array(pred_list)
  pred = pred.astype(np.bool)
  gt = np.array(gt_list)
  gt = gt.astype(np.bool)
  intersection = pred & gt
  sum_intersection = np.sum(intersection, axis=(1, 2))
  union = pred | gt
  denominator = np.sum(union, axis=(1, 2))

  valid = (denominator > 0)
  # When union is 0, defined IoU = 1
  iou = np.ones(sum_intersection.shape)
  iou[valid] = (sum_intersection[valid] / denominator[valid].astype(np.float))
  return iou


# bi_w = 4, bi_xy_std = 67, bi_rgb_std = 3, pos_w = 3, pos_xy_std = 1.
# Parameters from DeepLab
def apply_crf(prob, img, shape=(480, 854), bi_w=4, bi_xy_std=67, bi_rgb_std=3, pos_w=3, pos_xy_std=1):
  prob = np.maximum(prob, 1e-6)
  img = np.ascontiguousarray(img)
  prob = np.ascontiguousarray(prob.swapaxes(0, 2).swapaxes(1, 2))
  prob = prob.reshape(2, -1)

  height = shape[0]
  width= shape[1]
  d = dcrf.DenseCRF2D(width, height, 2)
  d.setUnaryEnergy((-np.log(prob)).astype(np.float32))
  d.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w)
  d.addPairwiseBilateral(sxy=bi_xy_std, srgb=bi_rgb_std, rgbim=img, compat=bi_w)
  processed = d.inference(5)
  seg = np.argmax(processed, axis=0).reshape(480, 854)
  return seg