import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shapely 
from shapely.geometry import MultiPoint, Polygon
from math import pi
from itertools import combinations
import os

EPS = 1e-5

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def homogenize (x) : 
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

def unhomogenize (x) : 
    return x[:, :-1] / x[:, -1:]

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

def find_quads_and_from_lines(a, b, c, d):
    line_sets = [[a, b, c, d], [a, d, b, c], [a, b, d, c]]
    line_sets_np = np.array(line_sets)
    line_sets_np = np.concatenate((line_sets_np, line_sets_np[:, :1, :]), axis=1)
    line_sets_np = np.concatenate((line_sets_np[:, :-1], line_sets_np[:, 1:]), axis=2)
    x = (line_sets_np[:, :, :2] * line_sets_np[:, :, 2:4]).sum(2)
    y = (line_sets_np[:, :, 4:6] * line_sets_np[:, :, 6:]).sum(2)
    #   0   1   2   3   4   5   6   7
    # px1 py1 nx1 ny1 px2 px2 nx2 ny2
    det = (line_sets_np[:, :, 2] * line_sets_np[:, :, 7] - line_sets_np[:, :, 3] * line_sets_np[:, :, 6])
    X = ( line_sets_np[:, :, 7] * x - line_sets_np[:, :, 3] * y) / det
    Y = (-line_sets_np[:, :, 6] * x + line_sets_np[:, :, 2] * y) / det
    intersections = np.stack((X, Y)).T.transpose((1, 0, 2))
    convex_mask = is_convex_poly_vec(intersections)
    intersections = intersections[convex_mask]
    quads = []
    for intersection in intersections : 
        poly = Polygon(intersection)
        quads.append(poly)
    return quads

def area_oriented_polygon (x) :
    """
    x is a tensor of shape [N, K, 2]
    """
    N, K, *_ = x.shape
    areas = []
    for i in range(K - 2) :
        A = x[:, 0:1, :]
        BC = x[:, i+1:i+3,:]
        ABC = np.concatenate((A, BC), axis=1)
        areas.append(np.abs(signed_area_triangle(ABC)))
    all_areas = np.stack(areas) # [K - 2, N]
    area = all_areas.sum(0) # [N]
    return area

def find_quads_from_lines_vec (a, b, c, d_s) : 
    N, L = d_s.shape
    a_s = np.tile(a[None, ...], (N, 1))
    b_s = np.tile(b[None, ...], (N, 1))
    c_s = np.tile(c[None, ...], (N, 1))
    CA = np.stack((a_s, b_s, c_s, d_s), axis=1) # [N, 4, 4] 
    CB = np.stack((a_s, d_s, b_s, c_s), axis=1) # [N, 4, 4] 
    CC = np.stack((a_s, b_s, d_s, c_s), axis=1) # [N, 4, 4] 
    line_sets_np = np.concatenate((CA, CB, CC), axis=0) # [3 * N, 4, 4]
    line_sets_np = np.concatenate((line_sets_np, line_sets_np[:, :1, :]), axis=1)
    line_sets_np = np.concatenate((line_sets_np[:, :-1], line_sets_np[:, 1:]), axis=2)
    x = (line_sets_np[:, :, :2] * line_sets_np[:, :, 2:4]).sum(2)
    y = (line_sets_np[:, :, 4:6] * line_sets_np[:, :, 6:]).sum(2)
    #   0   1   2   3   4   5   6   7
    # px1 py1 nx1 ny1 px2 px2 nx2 ny2
    det = (line_sets_np[:, :, 2] * line_sets_np[:, :, 7] - line_sets_np[:, :, 3] * line_sets_np[:, :, 6])
    X = ( line_sets_np[:, :, 7] * x - line_sets_np[:, :, 3] * y) / (det + 1e-7)
    Y = (-line_sets_np[:, :, 6] * x + line_sets_np[:, :, 2] * y) / (det + 1e-7)
    intersections = np.stack((X, Y)).T.transpose((1, 0, 2))
    convex_mask = is_convex_poly_vec(intersections)
    intersections = intersections[convex_mask]
    return intersections

def solve_2d_systems (A, b) : 
    """ 
    A - [N, 4] [a, b, c, d] corresponding to 
    [a b]
    [c d]

    [d -b]
    [-c a]
    
    inv [w x] 
        [y z]
    b - [N, 2]
    """ 
    det = A[:, 0] * A[:, 3] - A[:, 1] * A[:, 2]

    w = A[:, 3]
    x = -A[:, 1]
    y = -A[:, 2]
    z = A[:, 0]

    a_x = (w * b[:, 0] + x * b[:, 1]) / det
    a_y = (y * b[:, 0] + z * b[:, 1]) / det

    ans = np.stack((a_x, a_y)).T
    return ans

def barycentric_coordinates (triangles, points) : 
    """ 
    triangles  - [M, 3, 2] a, b, c
    points     - [N, 2]

    b - a and c - a are the basis vectors
    [ (b-a)_x (b-a)_y ] [1]
    [ (c_a)_x (c-a)_y ] [0]

    [ (b-a)_x (c-a)_x ] [1]
    [ (b_a)_y (c-a)_y ] [0]

    [ (b-a)_x (c-a)_x ] [alpha] = p - a
    [ (b_a)_y (c-a)_y ] [gamma]

    p - a = alpha (b - a) + beta (c - a) 
    p - a = alpha b + beta c + a (1 - alpha - beta)
    
    return - [M, N, 3]
    """
    M, *_ = triangles.shape
    N, *_ = points.shape
    assert M > 0 and N > 0, "Don't know what to do with empty arrays" 
    points = np.tile(points[None, ...], (M, 1, 1)) # [M, N, 2]
    triangles = np.tile(triangles[:, None, ...], (1, N, 1, 1)) # [M, N, 3, 2]
    points_minus_a = points - triangles[:, :, 0, :]
    bases_mat = (triangles[:, :, 1:, :] - triangles[:, :, :1, :]).transpose((0, 1, 3, 2)) # [M, N, 2, 2] 
    bases_mat = bases_mat.reshape(M, N, -1)
    alpha_beta = solve_2d_systems(bases_mat.reshape(-1, 4), points_minus_a.reshape(-1, 2)).reshape(M, N, 2)
    gamma = 1.0 - alpha_beta.sum(-1) # [M, N] 
    bary = np.concatenate((gamma.reshape(M, N, 1), alpha_beta), axis=-1)
    return bary

def points_inside_triangles (triangles, points) : 
    """
    triangles  - [M, 4, 2]
    points     - [N, 2]
    
    return - [M] mask of triangles that have all points inside them
    """ 
    bc = barycentric_coordinates(triangles, points) 
    mask = np.all(-EPS <= bc, axis=(1, 2)) & np.all(bc <= 1 + EPS, axis=(1, 2))
    return mask

def points_in_quads (quads, points) : 
    """ 
    quads  - [M, 4, 2]
    points - [N, 2]

    return - [M] mask of quads that have all points inside them
    """ 
    M, K, *_ = quads.shape
    N, *_ = points.shape
    assert M > 0 and N > 0, "Don't know what to do with empty arrays" 
    mask = np.zeros((M, N), dtype=bool)
    for i in range(K - 2) :
        A = quads[:, 0:1, :]
        BC = quads[:, i+1:i+3,:]
        ABC = np.concatenate((A, BC), axis=1)
        bc = barycentric_coordinates(ABC, points) # [M, N, 3]
        mask = mask | (np.all(0.0 <= bc, axis=2) & np.all(bc <= 1.0, axis=2))
    mask = np.all(mask, axis=1)
    return mask

def is_convex_poly (points) : 
    points = np.concatenate((points, points), axis=0)
    N = points.shape[0]
    pi = points[:-2]
    pj = points[1:-1]
    pk = points[2:]
    d1 = pj - pi
    d2 = pk - pj
    zs = (d1[:, 0] * d2[:, 1]) - (d1[:, 1] * d2[:, 0])
    return np.all(zs >= -1e-7) or np.all(zs <= 1e-7)

def is_convex_poly_vec (points) : 
    points = np.concatenate((points, points), axis=1)
    pi = points[:, :-2]
    pj = points[:, 1:-1]
    pk = points[:, 2:]
    d1 = pj - pi
    d2 = pk - pj
    zs = (d1[:, :, 0] * d2[:, :, 1]) - (d1[:, :, 1] * d2[:, :, 0])
    return np.all(zs >= -1e-7, axis=1) | np.all(zs <= 1e-7, axis=1)

def points_to_line(a, b):
    a_np = np.array(a)
    b_np = np.array(b)
    direction = b_np - a_np
    normal = np.array([-direction[1], direction[0]])
    normal = normal / np.linalg.norm(normal)
    return np.concatenate((a_np, normal))

def find_minimum_quad (point_cloud) : 
    """ 
    Algorithm based on Lemma 1 from: 

        Minimum area circumscribing Polygons -- Aggarwal et al. 

    which states that for a k-gon Q (k >= 4) to be a globally minimum 
    polygon that encloses P, atleast k-1 edges **flush** with P. 

    This suggests a dumb algorithm for finding the minimum quad 
    where we: 
        
        * Select any of the (N 3) edges and any point on the hull.
        * Draw 360 lines through the point and check if convex hull 
        points lie on one side of the line.
        * If not, discard point
        * Else, iterate over valid lines, compute the quad and keep 
        the one with minimum area.
    """
    point_set = MultiPoint(point_cloud)
    convex_hull_shapely = point_set.convex_hull
    convex_hull = np.array(convex_hull_shapely.exterior.coords)
    poly = convex_hull[:-1]
    ijk = list(combinations(range(len(convex_hull) - 1), 3))
    theta = np.linspace(0, pi + EPS, 180)
    normals = np.stack((np.cos(theta), np.sin(theta))).T # [N, 2]
    best_quad, best_area = None, None
    convex_hull_lines = [points_to_line(convex_hull[i], convex_hull[i + 1]) for i in range(len(convex_hull) - 1)]
    candidate_lines = [] 
    poly_sq = np.tile(poly[None, ...], (poly.shape[0], 1, 1))
    poly_sq_t = poly_sq.transpose((1, 0, 2))
    poly_minus_p = poly_sq - poly_sq_t # [N, N, 2]
    poly_minus_p_dot_normal = poly_minus_p @ (normals.T)
    valid_mask = np.all(poly_minus_p_dot_normal >= -EPS, axis=1) | np.all(poly_minus_p_dot_normal <= EPS, axis=1)
    for (i, j, k) in ijk: 
        valid_mask_ijk = np.copy(valid_mask)
        valid_mask_ijk[i:i+2] = False
        valid_mask_ijk[j:j+2] = False
        valid_mask_ijk[k:k+2] = False
        point_ids, normal_ids = np.where(valid_mask_ijk)
        candidate_lines = np.concatenate((poly[point_ids], normals[normal_ids]), axis=1)
        li = convex_hull_lines[i]
        lj = convex_hull_lines[j]
        lk = convex_hull_lines[k]
        np_quads = find_quads_from_lines_vec(li, lj, lk, candidate_lines)
        mask = points_in_quads(np_quads, poly)
        candidate_quads = np_quads[mask]
        areas = area_oriented_polygon(candidate_quads)
        if len(areas) > 0 :
            best_candidate = candidate_quads[np.argmin(areas)]
            if best_quad is None or (best_area > np.min(areas)): 
                best_quad = best_candidate
                best_area = np.min(areas)
    return Polygon(best_quad)

if __name__ == "__main__" : 
    fig, ax = plt.subplots(3, 3)
    for i in range(3):
        for j in tqdm(range(3)) : 
            while True: 
                try : 
                    pts = np.random.rand(1000, 2)
                    mat = 2 * np.random.randn(3, 3)
                    new_pts = unhomogenize((mat @ homogenize(pts).T).T)
                    print(new_pts.shape)
                    quad = np.array(find_minimum_quad(new_pts).exterior.coords)
                    ax[i][j].scatter(new_pts[:, 0], new_pts[:, 1])
                    ax[i][j].plot(quad[:, 0], quad[:, 1], c='r')
                    break
                except Exception :
                    pass

    plt.show()

