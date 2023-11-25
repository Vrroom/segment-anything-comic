""" 
This file contains some losses and some metrics 
"""
import time
import random
import torch
from torchTools import * 
from matching import *
from copy import deepcopy
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate

def signed_area_triangle (x) : 
    """ 
    x is a tensor of shape [N, 3, 2]
    """ 
    pB = x[:, 1, :] - x[:, 0, :] # [N, 2]
    pC = x[:, 2, :] - x[:, 0, :] # [N, 2]

    xB = pB[:, 0]
    yB = pB[:, 1]

    xC = pC[:, 0]
    yC = pC[:, 1]

    return 0.5 * (xB * yC - xC * yB) # [N]

def signed_area_oriented_polygon (x) : 
    """ 
    x is a tensor of shape [N, K, 2]
    """
    N, K, *_ = x.shape
    areas = []
    for i in range(K - 2) : 
        A = x[:, 0:1, :] 
        BC = x[:, i+1:i+3,:] 
        ABC = torch.cat((A, BC), 1)
        areas.append(signed_area_triangle(ABC))
    all_areas = torch.stack(areas) # [K - 2, N]
    area = all_areas.sum(0) # [N] 
    return area

def rel_orientation_loss (x, y) : 
    """ 
    x - [N, 4, 2]
    y - [N, 4, 2]
    """ 
    x_cyc = torch.cat((x, x[:, :1, :]), 1)
    y_cyc = torch.cat((y, y[:, :1, :]), 1)

    x_deltas = x_cyc[:, 1:, :] - x_cyc[:, :-1, :] 
    y_deltas = y_cyc[:, 1:, :] - y_cyc[:, :-1, :] 

    x_del_norm = unitNorm(x_deltas, 2) 
    y_del_norm = unitNorm(y_deltas, 2) 

    dot_product = (x_del_norm * y_del_norm).sum(2) 

    one_minus_dot_product = (1 - dot_product).max(1)[0]

    loss = one_minus_dot_product.mean()

    return loss

def matched_l1_metric(x, y):
    """
    x - [L1, 2] # A polygon
    y - [L2, 2] # Another polygon

    Here, we first find the best match between points in x and y
    using bipartite matching and then compute the l1 distance over
    the optimal match
    """
    L1 = x.shape[0]
    L2 = y.shape[0]

    x_full = x.unsqueeze(1).repeat(1, L2, 1) 
    y_full = y.unsqueeze(0).repeat(L1, 1, 1)

    pair_wise_l1 = (x_full - y_full).abs().sum(-1) # [L1, L2]


    row_idx = torch.arange(L1).unsqueeze(0).repeat(L2, 1).reshape(-1) 
    col_idx = torch.arange(L2).unsqueeze(0).repeat(L1, 1).t().reshape(-1)

    costTable = dict(zip(zip(row_idx.tolist(), col_idx.tolist()), pair_wise_l1[row_idx, col_idx]))
    opt_score = bestAssignmentCost(costTable) / min(L1, L2)

    return opt_score

def polygon_iou (x, y) : 
    """
    x - [L, 2] # A polygon
    y - [L, 2] # Another polygon

    Here, we first find the best match between points in x and y
    using bipartite matching and then compute the l1 distance over
    the optimal match
    """
    px = Polygon(x)
    py = Polygon(y) 
    int_area = px.intersection(py).area
    uni_area = px.union(py).area
    return int_area / uni_area

def pck_at_alpha(x, y, alpha, scale):
    L1 = x.shape[0]
    L2 = y.shape[0]

    x_full = x.unsqueeze(1).repeat(1, L2, 1) 
    y_full = y.unsqueeze(0).repeat(L1, 1, 1)

    pair_wise_l2 = torch.sqrt(((x_full - y_full) ** 2).sum(-1)) # [L1, L2]

    row_idx = torch.arange(L1).unsqueeze(0).repeat(L2, 1).reshape(-1) 
    col_idx = torch.arange(L2).unsqueeze(0).repeat(L1, 1).t().reshape(-1)

    costTable = dict(zip(zip(row_idx.tolist(), col_idx.tolist()), pair_wise_l2[row_idx, col_idx]))
    matching = optimalBipartiteMatching(costTable)

    threshold = alpha * scale
    correct = 0 
    for (i, j) in matching.items() : 
        correct += int(pair_wise_l2[i, j] <= threshold)

    total_points = max(L1, L2)
    return (correct / total_points)

if __name__ == "__main__" : 
    """ test case for relative orientation loss """ 
    x = torch.tensor([[0., 0.], [0., 10.0], [5.0, 10.0], [5.0, 0.0]]).unsqueeze(0)
    print(rel_orientation_loss(x, x))
    """ test case for matched l1 metric """ 
    x_ = x.squeeze() 
    idx = list(range(4))
    random.shuffle(idx)
    y_ = deepcopy(x_) 
    y_ = y_[idx, ...]
    print(matched_l1_metric(x_, y_))
    print(polygon_iou(x_, x_))
    print(pck_at_alpha(x_, x_, 0.1, 1.0))
