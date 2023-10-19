import pytorch_lightning as pl
import skimage
from imageOps import *
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
from einops import rearrange
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
from torchvision import datasets, transforms
import os
from osTools import *
from PIL import Image, ImageDraw, ImageFilter
import random
from segment_anything.utils.transforms import *
from segment_anything import SamPredictor, sam_model_registry, apply_transform_to_pil_without_sam_model, unnormalize_tensor
from torchTools import *
from args import *
import math
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate
from more_itertools import flatten
from logTools import *

def config_plot(ax):
    """ Function to remove axis tickers and box around a given axis """
    ax.set_frame_on(False)
    ax.axis('off')

""" Stolen from https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon """
def sample_random_points_in_polygon(shape, k=1):
    "Return list of k points chosen uniformly at random inside the polygon."
    polygon = Polygon(shape)
    areas = []
    transforms = []
    for t in triangulate(polygon):
        areas.append(t.area)
        (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
        transforms.append([x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0])
    points = []
    for transform in random.choices(transforms, weights=areas, k=k):
        x, y = [random.random() for _ in range(2)]
        if x + y > 1:
            p = Point(1 - x, 1 - y)
        else:
            p = Point(x, y)
        points.append(affine_transform(p, transform))
    return [p.coords for p in points]

def sorted_points(points):
    # sort points based on distance to origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)

    # pick the first point
    first_point = points[0]
    x, y = first_point

    # used to sort points based on the angle subtended on first point
    def angle_key(point):
        px, py = point[0] - x, point[1] - y
        theta = math.atan2(py, px)
        if theta < 0 :
            theta += 2 * math.pi
        return theta

    pts = [first_point] + sorted(points[1:], key=angle_key)
    return pts

def composite_mask(image, mask, alpha=0.2):
    image = skimage.transform.resize(image, mask.shape, preserve_range=True).astype(np.uint8)
    white = [255, 255, 255]
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 1] = white
    composite = np.uint8(image * (1 - alpha) + mask_rgb * alpha)
    return composite

def visualize_batch (sam_model, batch, dataset, outputs=None, save_to=None) : 
    """ This visualized a batch from the dataset """ 
    batch = tensorApply(batch, lambda x: x.to(torch.device('cuda')))
    # extract stuff from batch

    if 'features' in batch : 
        features = batch['features']
    else : 
        assert 'img' in batch, "Either image or features needed for this" 
        with torch.no_grad() : 
            features = sam_model.image_encoder(batch['img'])

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
    n_masks = len(masks) 
    best_masks = []
    # process and select best masks
    for i, mask in enumerate(masks) : 
        mask_threshed = (mask > sam_model.mask_threshold).squeeze()
        best_masks.append(mask_threshed[torch.argmax(iou_predictions[i])].detach().cpu().numpy()) 

    plots = []
    for i in range(n_masks) :
        fig, ax = plt.subplots(1, 1)
        mask_to_show = best_masks[i]

        if outputs is not None: 
            # Plot predictions
            pred_pts = normalized_point_to_image_point(outputs['pred'][i], input_size[i], original_size[i]).detach().cpu().numpy()
            ax.scatter(pred_pts[:, 0], pred_pts[:, 1], c=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)], marker='x', alpha=0.5)

        # Plot GT points
        pts = normalized_point_to_image_point(shape[i], input_size[i], original_size[i]).detach().cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], c=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)], alpha=0.5)

        # Plot Query Point
        sample_point = model_point_to_image_point(point_coords[i], input_size[i], original_size[i]).detach().cpu().numpy()
        ax.scatter(sample_point[:, 0], sample_point[:, 1], c='g') 

        # handle the case where the image is provided in the batch
        if 'img' in batch : 
            h, w = input_size[i]
            img = unnormalize_tensor(batch['img'][i])
            vis_img = (255. * normalize2UnitRange(img).permute(1,2,0).detach().cpu().numpy()[:h, :w]).astype(np.uint8) 
        else : 
            vis_img = np.array(Image.open(f'{dataset.folders[index[i]]}/img.png'))

        vis_img = composite_mask(vis_img, mask_to_show.astype(np.uint8))
        ax.imshow(vis_img)

        # remove ticks and box
        config_plot(ax)

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

    def __init__(self, folders_list, target_img_size=1024, precompute_features=True):
        self.folders = folders_list
        self.target_img_size = target_img_size
        self.precompute_features = precompute_features
        if self.precompute_features : 
            self.features = torch.cat([torch.load(osp.join(_, 'vit_h_features.pt'), map_location='cpu') for _ in self.folders])
        self.pil_paths = [osp.join(_, 'img.png') for _ in self.folders]
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
        if self.precompute_features : 
            features = self.features[i] 
        else : 
            img = apply_transform_to_pil_without_sam_model(Image.open(self.pil_paths[i]), 'cpu').squeeze()

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
    

        if self.precompute_features : 
            return dict(
                features=features,            # [256, 64, 64]
                point_coords=point_coords,    # [1, 2] 
                point_labels=point_labels,    # [1]
                original_size=original_size,  # [2]
                input_size=input_size,        # [2]
                shape=shape,                  # [4, 2], float, [-1.0, 1.0]
                index=torch.tensor([i])       # [1]
            )
        else : 
            # now the model training code will compute features. We'll just give the image
            return dict(
                img=img,                      # [3, 1024, 1024]
                point_coords=point_coords,    # [1, 2] 
                point_labels=point_labels,    # [1]
                original_size=original_size,  # [2]
                input_size=input_size,        # [2]
                shape=shape,                  # [4, 2], float, [-1.0, 1.0]
                index=torch.tensor([i])       # [1]
            )

def transpose_points (pts) : 
    assert is_iterable(pts), '(transpose_points): I need an iterable'
    if all(isinstance(_, int) for _ in pts) : 
        assert len(pts) == 2, f'(transpose_points): I don\'t know what to do with {len(pts)}-D point' 
        x, y = pts
        return y, x
    else : 
        return [transpose_points(_) for _ in pts]

def transpose_simple_comic_layout_data (data) : 
    return {
        'img': data['img'].rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM),
        'original_size': transpose_points(data['original_size']),
        'boxes': fix_boxes(data['boxes']) 
        # NOTE: ^ this function is wrongly named in this context. 
        # There is nothing wrong with these boxes. The aim is to 
        # simply transpose (switch x and y coordinates).
    }

def generate_simple_comic_layout():
    while True :
        # Choose an aspect ratio
        aspect_ratios = [
            {"width": 1, "height": 1},
            {"width": 4, "height": 3},
            {"width": 16, "height": 9},
            {"width": 21, "height": 9},
            {"width": 3, "height": 2},
            {"width": 9, "height": 16},
            {"width": 2.35, "height": 1},
            {"width": 1.85, "height": 1},
            {"height": 4, "width": 3},
            {"height": 16, "width": 9},
            {"height": 21, "width": 9},
            {"height": 3, "width": 2},
            {"height": 9, "width": 16},
            {"height": 2.35, "width": 1},
            {"height": 1.85, "width": 1}
        ]
        chosen_ratio = random.choice(aspect_ratios)

        # Set image dimensions
        if chosen_ratio["width"] > chosen_ratio["height"]:
            width = 1024
            height = int(width / chosen_ratio["width"] * chosen_ratio["height"])
        else:
            height = 1024
            width = int(height * chosen_ratio["width"] / chosen_ratio["height"])

        # Create an image with background color
        background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        img = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(img)

        # Border settings
        border_thickness = random.randint(1, 20)
        border_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Whether to fill the box
        box_fill = None if random.choice([True, False]) else (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Whether to draw rectangle or rounded rectangle
        draw_rect = random.random() > 0.25
        rect_radius = random.randint(1, 50)

        # Gutter settings
        gutter = random.choice([True, False])
        gutter_width = random.randint(1, 50) if gutter else 0

        # Margin settings
        margin_x = random.randint(0, width // 10)
        margin_y = random.randint(0, height // 10)

        # Rows and Columns
        rows = random.choice([1, 2, 3, 4])
        row_height = (height - 2 * margin_y - (rows - 1) * gutter_width) // rows

        y_start = margin_y
        boxes = []
        for _ in range(rows):
            boxes.append([])
            cols = random.choice([1, 2, 3, 4])
            col_width = (width - 2 * margin_x - (cols - 1) * gutter_width) // cols

            x_start = margin_x
            for _ in range(cols):
                boxes[-1].append((x_start, x_start + col_width, y_start, y_start + row_height))
                if draw_rect : 
                    draw.rectangle(
                        [(x_start, y_start), (x_start + col_width, y_start + row_height)], 
                        fill=box_fill, 
                        outline=border_color, 
                        width=border_thickness
                    )
                else : 
                    draw.rounded_rectangle(
                        [(x_start, y_start), (x_start + col_width, y_start + row_height)], 
                        radius=rect_radius, 
                        fill=box_fill, 
                        outline=border_color, 
                        width=border_thickness
                    )
                x_start += col_width + gutter_width

            y_start += row_height + gutter_width

        # Apply gaussian blur so that not overly dependent on sharp edges
        apply_gaussian_blur = random.random() > 0.25
        kernel_size = random.choice([2,3,4,5]) 
        if apply_gaussian_blur : 
            img = img.filter(ImageFilter.GaussianBlur(kernel_size))

        data = {
            'img': img, 
            'original_size': tuple(reversed(img.size)),
            'boxes': list(flatten(boxes))
        }

        # Randomly transpose rows and columns for added flair
        transpose_data = random.choice([True, False])
        if transpose_data : 
            data = transpose_simple_comic_layout_data(data) 

        yield data

class RandomComicLayoutDataset (Dataset) : 

    def __init__(self, random_gen_len=10000, target_img_size=1024):
        self.random_gen_len = random_gen_len
        self.target_img_size = target_img_size
        self.transform = ResizeLongestSide(target_img_size)
        self.generator = generate_simple_comic_layout()

    def __len__(self):
        return self.random_gen_len # len(self.folders)

    def __getitem__(self, i):
        # get features
        data = next(self.generator)
        img = apply_transform_to_pil_without_sam_model(data['img'], 'cpu').squeeze()

        # get original and transformed image sizes
        original_size = data['original_size'] 
        input_size = original_size_to_input_size(self.transform, original_size)

        boxes = data['boxes']
        N = len(boxes)
        shape_id = random.randint(0, N - 1)

        # prepare the shape
        shape = box_to_shape(boxes[shape_id])

        # now sample random points from the corresponding box
        point_coords = sample_random_points_in_polygon(shape, 1)[0]
        point_coords = torch.from_numpy(self.transform.apply_coords(np.array(point_coords).astype(np.float32), original_size)) # [1, 2]

        shape = torch.from_numpy(self.transform.apply_coords(np.array(shape).astype(np.float32), original_size))
        # normalize the shape
        shape = (2.0 * (shape / self.target_img_size) - 1.0).float()

        # all the query points are foreground in our case
        point_labels = torch.ones((1,)).float()

        # cast to tensor 
        original_size = torch.tensor(original_size)
        input_size = torch.tensor(input_size)

        # now the model training code will compute features. We'll just give the image
        return dict(
            img=img,                      # [3, 1024, 1024]
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
        self.precompute_features = args.precompute_features
        self.files = deterministic_shuffle(list_base_dir(self.base_dir))
        self.train_files, self.test_files = split_train_test(self.files, 0.9)

    def setup(self, stage=None):
        if self.precompute_features : 
            self.train_data = FrameDataset(self.train_files, precompute_features=self.precompute_features) 
            self.test_data = FrameDataset(self.test_files, precompute_features=self.precompute_features)
        else : 
            print('Using two datasets') 
            self.train_data = ConcatDataset([
                FrameDataset(self.train_files, precompute_features=self.precompute_features),
                RandomComicLayoutDataset()
            ])
            self.test_data = ConcatDataset([
                FrameDataset(self.train_files, precompute_features=self.precompute_features),
                RandomComicLayoutDataset(random_gen_len=100) 
            ])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__" : 
    # test with precomputed_features=True
    seed = 2
    seed_everything(seed)
    datamodule = FrameDataModule(DictWrapper(dict(base_dir='../comic_data', batch_size=4, num_workers=0, precompute_features=True)))
    datamodule.setup()
    for batch in datamodule.train_dataloader() : 
        break
    print(batch.keys())
    for k in batch.keys() :
        print(k, batch[k].shape)
    sam_model = sam_model_registry["vit_h"](checkpoint="./checkpoints/sam_vit_h_4b8939.pth").cuda()
    visualize_batch(sam_model, batch, datamodule.train_data, save_to='img1.png')

    # test with precomputed_features=False
    seed_everything(seed)
    datamodule = FrameDataModule(DictWrapper(dict(base_dir='../comic_data', batch_size=4, num_workers=0, precompute_features=False)))
    datamodule.setup()
    for batch in datamodule.train_dataloader() : 
        break
    print(batch.keys())
    for k in batch.keys() :
        print(k, batch[k].shape)
    visualize_batch(sam_model, batch, datamodule.train_data, save_to='img2.png')
