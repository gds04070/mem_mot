import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter
import time

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )
    return ious

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix

def calculate_fuse_distance(tracks, detections, fuse, term_type, iou_weight = 1, reid_weight=0.5, distance_weight=0.5):
    iou_cost_matrix = iou_distance(tracks, detections) * iou_weight
    if fuse:
        iou_cost_matrix = fuse_score(iou_cost_matrix, detections)
    feature_cost_matrix, distance_cost_matrix = memory_distance(tracks, detections, term_type)
    # distance_cost_matrix= 1 - 1 / (1+ distance_cost_matrix**0.25)
    distance_cost_matrix = 1 - 1/(1.1**distance_cost_matrix) # best
    feature_cost_matrix *= reid_weight
    distance_cost_matrix *= distance_weight
    cost_matrix = iou_cost_matrix + (feature_cost_matrix + distance_cost_matrix) * (1 - iou_weight)
    return cost_matrix

def memory_distance(tracks, detections, term_type): # tracks: M (memory len - l), detections: N
    def calculate_weight(l, method):
        if method == 'constant':
            w = np.ones(l)
        elif method == 'linear': # linear
            if l == 1:
                return [1.]
            w = np.linspace(0, 1, l)
        else: # exp
            w = np.exp(np.linspace(0, 1, l))
            w = w / np.max(w)
        return w

    if term_type == 'half':
        term = 2
    else:
        term = 1

    distance_cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    feature_cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if distance_cost_matrix.size == 0:
        return feature_cost_matrix, distance_cost_matrix # M x N
    detection_features = np.array([det.curr_feature for det in detections]) # N
    detection_tlbrs = np.array([det.tlbr for det in detections]) # N
    for ti, curr_track in enumerate(tracks): # iter M
        memory_length=len(curr_track.memory['features'])//term
        curr_track_features = np.array(curr_track.memory['features'])[memory_length-1:] # l
        curr_track_tlbrs = np.array(curr_track.memory['tlbrs'])[memory_length-1:] # l
        # N x l
        feature_cost = features_distance(detection_features, curr_track_features) # N x l
        tlbr_cost = features_distance(detection_tlbrs, curr_track_tlbrs, 'euclidean') # N x l
        # /l -> N
        # feature_cost = np.mean(feature_cost, axis=1) # N | np.max(feature_cost, axis=1) | np.average()
        feature_cost = np.average(feature_cost, axis=1, weights=calculate_weight(len(curr_track_features), 'linear'))
        tlbr_cost = np.average(tlbr_cost, axis=1, weights=calculate_weight(len(curr_track_tlbrs), 'linear'))
        # [M, N]
        feature_cost_matrix[ti] = feature_cost
        distance_cost_matrix[ti] = tlbr_cost
    return feature_cost_matrix, distance_cost_matrix

def features_distance(target, features, metric='cosine'):
    cost_matrix = np.zeros((len(target), len(features)), dtype = np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    if not isinstance(target, np.ndarray):
        target = np.asarray(target)
    if not isinstance(features, np.ndarray):
        features = np.asarray(features)
    if len(target.shape) == 1:
        target = target.reshape(1, -1)
    cost_matrix = np.maximum(0.0, cdist(target, features, metric))
    return cost_matrix

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

# TODO
def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float) # !!!!
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix