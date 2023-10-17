import pytorch_lightning as pl
from imageOps import *
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
from einops import rearrange
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import os
from osTools import *
from PIL import Image
import random
from segment_anything.utils.transforms import *
from segment_anything import SamPredictor, sam_model_registry
from torchTools import *
from args import *
import math

def sorted_points(points):
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    first_point = points[0]

    cx, cy = sum(x for x, y in points) / len(points), sum(y for x, y in points) / len(points)

    p = first_point[0] - cx, first_point[1] - cy
    p_perp = -p[1], p[0]

    cb = np.linalg.inv(np.array([[p[0], p_perp[0]], [p[1], p_perp[1]]]))

    def angle_key(point):
        p = point[0] - cx, point[1] - cy
        x, y = (cb @ np.array(p).reshape(-1, 1)).reshape(-1)
        theta = math.atan2(y, x)
        if theta < 0 :
            theta += 2 * math.pi
        return theta

    return sorted(points, key=angle_key)

def visualize_batch (sam_model, batch, dataset, outputs=None, save_to=None) : 
    """ This visualized a batch from the dataset """ 
    batch = tensorApply(batch, lambda x: x.to(torch.device('cuda')))
    # extract stuff from batch
    features = batch['features']
    point_coords = batch['point_coords']
    point_labels = batch['point_labels']
    original_size = batch['original_size']
    input_size = batch['input_size']
    shape = batch['shape']
    index = batch['index']

    if outputs is None:
        points = (point_coords, point_labels)

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
        
        # Predict masks
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=features,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            interleave=False, # this ensures correct behaviour when each prompt is for a different image
        )
    else :
        low_res_masks, iou_predictions = outputs['low_res_masks'], outputs['iou_predictions']

    # Upscale the masks to the original image resolution
    masks = sam_model.postprocess_masks_size_list(low_res_masks, input_size, original_size)
    masks = torch.cat(masks)
    masks = masks > sam_model.mask_threshold
    n_masks = masks.shape[0]
    best_masks = masks[torch.arange(n_masks), torch.argmax(iou_predictions, 1)].detach().cpu().numpy()

    plots = []
    for i in range(n_masks) :
        fig, (ax1, ax2) = plt.subplots(1, 2)
        mask_to_show = best_masks[i]
        pts = normalized_point_to_image_point(shape[i], input_size[i], original_size[i]).detach().cpu().numpy()

        if outputs is not None: 
            pred_pts = normalized_point_to_image_point(outputs['pred'][i], input_size[i], original_size[i]).detach().cpu().numpy()
            ax1.scatter(pred_pts[:, 0], pts[:, 1], c='b', marker='x')

        ax1.scatter(pts[:, 0], pts[:, 1], c='r')
        sample_point = model_point_to_image_point(point_coords[i], input_size[i], original_size[i]).detach().cpu().numpy()
        ax1.scatter(sample_point[:, 0], sample_point[:, 1], c='g') 
        ax1.imshow(mask_to_show)
        ax2.imshow(Image.open(f'{dataset.folders[index[i]]}/img.png'))
        plots.append(fig_to_pil(fig))
        plt.close(fig)

    if save_to is not None: 
        make_image_grid(plots, False).save(save_to)
    else :
        plt.imshow(make_image_grid(plots, False))
        plt.show()

def fig_to_pil (fig) : 
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return Image.fromarray(data)

def box_to_shape(box) : 
    x, X, y, Y = box
    return sorted_points([(x, y), (x, Y), (X, Y), (X, y)])

def normalized_point_to_image_point (pt, input_size, original_size) : 
    target_img_size = int(max(input_size))
    factor = max(original_size) / max(input_size)
    pt = factor * target_img_size * ((pt / 2.0) + 0.5)
    return pt

def model_point_to_image_point(pt, input_size, original_size) : 
    factor = max(original_size) / max(input_size)
    pt = factor * pt
    return pt

def original_size_to_input_size(transform, original_size): 
    """ convert original size to the size seen by the model """
    input_size_np = transform.apply_coords(np.array((original_size,)), original_size)
    input_size_rounded = [round(x) for x in input_size_np.tolist()[0]]
    return tuple(input_size_rounded)

def correct_box (box) : 
    """ X should be along image width and Y should be along image height """
    x, X, y, Y = box
    return (y, Y, x, X)

def correct_point (point) : 
    """ X should be along image width and Y should be along image height """
    y, x = point
    return x, y

def fix_boxes (boxes) : 
    boxes = [correct_box(_) for _ in boxes]
    return boxes

def sample_random_point_in_box (box) : 
    x, X, y, Y = box
    return (random.randint(x, X), random.randint(y, Y))

def fix_points (shapes) : 
    shapes = [sorted([correct_point(_) for _ in shape]) for shape in shapes]
    return shapes

def deterministic_shuffle(lst, seed=0):
    random.seed(seed)
    random.shuffle(lst)
    return lst

def list_base_dir(base_dir):
    """ lists one dir down in a directory """
    result = []
    for root, dirs, _ in os.walk(base_dir):
        for d in dirs:
            path = os.path.join(root, d)
            subfolders = [os.path.join(path, sub) for sub in os.listdir(path)]
            result.extend(subfolders)
        break
    return result

def split_train_test (data, train_percent) : 
    train_size = int(len(data) * train_percent)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

class FrameDataset(Dataset):

    def __init__(self, folders_list, target_img_size=1024):
        self.folders = folders_list
        self.target_img_size = target_img_size
        self.features = torch.cat([torch.load(osp.join(_, 'vit_h_features.pt'), map_location='cpu') for _ in self.folders])
        self.img_sizes = [np.array(Image.open(osp.join(_, 'img.png'))).shape[:2] for _ in self.folders]
        self.transform = ResizeLongestSide(target_img_size)
        # now load the data
        self.data = []
        for base_path in self.folders: 
            with open(osp.join(base_path, 'data.pkl'), 'rb') as fp :
                self.data.append(pickle.load(fp))
        # fix boxes and shapes
        for i in range(len(self.data)) :
            # TODO: Visualize whether box and shapes are identical
            self.data[i]['boxes'] = fix_boxes(self.data[i]['boxes'])
            self.data[i]['shapes'] = [box_to_shape(_) for _ in self.data[i]['boxes']] # fix_points(self.data[i]['shapes'])

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, i):
        # get features
        features = self.features[i] 

        # get original and transformed image sizes
        original_size = self.img_sizes[i]
        input_size = original_size_to_input_size(self.transform, original_size)

        # pick a random shape id
        N = len(self.data[i]['shapes'])
        shape_id = random.randint(0, N - 1)

        # prepare the shape
        shape = self.data[i]['shapes'][shape_id]
        shape = torch.from_numpy(self.transform.apply_coords(np.array(shape).astype(np.float32), original_size))
        # normalize the shape
        shape = (2.0 * (shape / self.target_img_size) - 1.0).float()

        # now sample random points from the corresponding box
        point_coords = [sample_random_point_in_box(self.data[i]['boxes'][shape_id])] 
        point_coords = torch.from_numpy(self.transform.apply_coords(np.array(point_coords).astype(np.float32), original_size)) # [1, 2]

        # all the query points are foreground in our case
        point_labels = torch.ones((1,)).float()

        # cast to tensor 
        original_size = torch.tensor(original_size)
        input_size = torch.tensor(input_size)

        return dict(
            features=features,            # [256, 64, 64]
            point_coords=point_coords,    # [1, 2] 
            point_labels=point_labels,    # [1]
            original_size=original_size,  # [2]
            input_size=input_size,        # [2]
            shape=shape,                  # [4, 2], float, [-1.0, 1.0]
            index=torch.tensor([i])       # [1]
        )

class FrameDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.base_dir = args.base_dir
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.files = deterministic_shuffle(list_base_dir(self.base_dir))
        self.train_files, self.test_files = split_train_test(self.files, 0.9)

    def setup(self, stage=None):
        self.train_data = FrameDataset(self.train_files) 
        self.test_data = FrameDataset(self.test_files)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__" : 
    seed_everything(0)
    datamodule = FrameDataModule(DictWrapper(dict(base_dir='../comic_data', batch_size=8)))
    datamodule.setup()
    for batch in datamodule.train_dataloader() : 
        break
    print(batch.keys())
    for k in batch.keys() :
        print(k, batch[k].shape)
    sam_model = sam_model_registry["vit_h"](checkpoint="./checkpoints/sam_vit_h_4b8939.pth").cuda()
    visualize_batch(sam_model, batch, datamodule.train_data, save_to='img.png')

