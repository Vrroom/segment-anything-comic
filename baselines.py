from itertools import product, combinations
import cv2
from tqdm import tqdm
import pickle
import numpy as np
from PIL import Image
import imageio
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.measure import label
from datamodule import *
from matching import *
from losses import *
from imageOps import *
from model import *
from copy import deepcopy 
import pandas as pd
from pepperAndCarrotTools import *

transform = ResizeLongestSide(1024)

def halford (img) : 
    # this algorithm is taken from https://maxhalford.github.io/blog/comic-book-panel-segmentation/
    im = np.array(img.convert('RGB'))
    grayscale = rgb2gray(im)
    edges = canny(grayscale)
    thick_edges = dilation(dilation(edges)).astype(int)
    segmentation = ndi.binary_fill_holes(thick_edges)
    labels = label(segmentation)
    shapes = []
    for i in range(labels.max() + 1) :
        y_pt, x_pt = np.where(labels == i) 
        x, X = x_pt.min(), x_pt.max()
        y, Y = y_pt.min(), y_pt.max()
        shapes.append([(x, y), (X, y), (X, Y), (x, Y)])
    return shapes

def find_best_shape_matching (shape_set_1, shape_set_2, metric, mode='minimum') : 
    """
    We assume that shape metrics are in range [0, 1].
    """
    assert mode in ['minimum', 'maximum'], "Mode has to be either \'minimum\' or \'maximum\'"
    costTable = dict()
    for i in range(len(shape_set_1)) : 
        for j in range(len(shape_set_2)) : 
            x = shape_set_1[i]
            y = shape_set_2[j]
            if mode == 'minimum': 
                costTable[(i, j)] = metric(x, y)
            else :
                costTable[(i, j)] = 1.0 - metric(x, y) 
    match = optimalBipartiteMatching(costTable)
    cost = sum([costTable[e] for e in match.items()]) / len(match)
    if mode == 'maximum' : 
        cost = 1.0 - cost
    return match, cost

def sam (img) :
    pass

def transform_shape (transform, shape, original_size, target_img_size) :
    shape = torch.from_numpy(transform.apply_coords(np.array(shape).astype(np.float32), original_size))
    shape = (2.0 * (shape / target_img_size) - 1.0).float()
    return shape


def draw_shape_on_image (img, shape, input_size, original_size) : 
    pts = normalized_point_to_image_point(shape, input_size, original_size).detach().cpu().numpy().astype(int)
    # Make visualization
    for point in pts:
        cv2.drawMarker(img, point, [255, 0, 0], markerType=5, markerSize=20, thickness=5)
    return img

def visualize_shape_matching (data, shape_set_1, shape_set_2, match) : 
    """ 
    data is a dictionary containing the image, original size and the original shape
    """
    img = data['img'] 
    original_size = data['original_size'] 
    input_size = original_size_to_input_size(transform, original_size)

    imgs = [] 
    for i, j in match.items() : 
        img_1 = np.array(deepcopy(img))
        img_2 = np.array(deepcopy(img))

        shape_1 = shape_set_1[i]
        shape_2 = shape_set_2[j] 

        img_1 = draw_shape_on_image(img_1, shape_1, input_size, original_size)
        img_2 = draw_shape_on_image(img_2, shape_2, input_size, original_size)

        img_1 = Image.fromarray(img_1)
        img_2 = Image.fromarray(img_2)

        imgs.append([img_1, img_2])

    img = make_image_grid(imgs)
    return aspectRatioPreservingResizePIL(img, 512)

def metrics_aggregator(*args):
    """
    Takes in any number of lists and returns a DataFrame with mean and std for each list.

    Args:
    *args: Lists of numeric values.

    Returns:
    DataFrame: A dataframe with the mean and std for each list.
    """

    means = [np.mean(arg) for arg in args]
    stds = [np.std(arg) for arg in args]
    data = { 'mean': means, 'std': stds }
    df = pd.DataFrame(data)
    return df

def evaluate_metrics_using_generator (generator) : 
    model = load_model('lightning_logs/version_24')

    seed = 1000
    seed_everything(seed)

    halford_matched_l1_scores, halford_polygon_iou_scores, halford_pck_at_alpha_scores, halford_fraction_matched = [], [], [], []
    ours_matched_l1_scores, ours_polygon_iou_scores, ours_pck_at_alpha_scores = [], [], []

    i = 0
    for data in tqdm(generator):
        if i > 1000 : break

        img = data['img'] 

        np_img = np.array(img)
        features = model.encode_image(np_img)

        original_size = data['original_size'] 

        point_samples = [list(sample_random_points_in_polygon(shape, 1)[0]) for shape in data['shapes']]

        original_shapes = [transform_shape(transform, _, original_size, 1024) for _ in data['shapes']]
        halford_shapes = [transform_shape(transform, _, original_size, 1024) for _ in halford(img)]
        ours_shapes = [transform_shape(transform, model.run_inference_simple(np_img, pt, features=features), original_size, 1024) for pt in point_samples]

        match, score = find_best_shape_matching(original_shapes, halford_shapes, matched_l1_metric)
        halford_matched_l1_scores.append(score)
        halford_fraction_matched.append(len(match) / len(halford_shapes))

        match, score = find_best_shape_matching(original_shapes, halford_shapes, polygon_iou, 'maximum')
        halford_polygon_iou_scores.append(score)

        match, score = find_best_shape_matching(original_shapes, halford_shapes, lambda x, y: pck_at_alpha(x, y, 0.1, 2*np.sqrt(2)), 'maximum')
        halford_pck_at_alpha_scores.append(score)

        match, score = find_best_shape_matching(original_shapes, ours_shapes, matched_l1_metric)
        ours_matched_l1_scores.append(score)

        match, score = find_best_shape_matching(original_shapes, ours_shapes, polygon_iou, 'maximum')
        ours_polygon_iou_scores.append(score)

        match, score = find_best_shape_matching(original_shapes, ours_shapes, lambda x, y: pck_at_alpha(x, y, 0.1, 2*np.sqrt(2)), 'maximum')
        ours_pck_at_alpha_scores.append(score)

        i += 1

    print('Halford')
    print(metrics_aggregator(halford_matched_l1_scores, halford_polygon_iou_scores, halford_pck_at_alpha_scores, halford_fraction_matched))
    print('Ours')
    print(metrics_aggregator(ours_matched_l1_scores, ours_polygon_iou_scores, ours_pck_at_alpha_scores))

if __name__ == "__main__" : 
    synthetic_generator = generate_simple_comic_layout()
    pc_generator = pepper_and_carrot_generator('../pepper_and_carrot_imgs/')

    evaluate_metrics_using_generator(pc_generator)
    evaluate_metrics_using_generator(synthetic_generator)
