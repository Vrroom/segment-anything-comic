from osTools import *
import hashlib
from sklearn.cluster import MeanShift
from losses import rel_orientation_loss
import cv2
from logTools import *
from PIL import Image
from args import *
import pickle
import pytorch_lightning as pl
import torch
from torch import nn
from segment_anything import SamPredictor, sam_model_registry, apply_transform_to_pil_without_sam_model
from segment_anything.modeling.mask_decoder import MLP
from segment_anything.utils.transforms import *
import torch.nn.functional as F
from itertools import chain
from copy import deepcopy 
import re

COLORS = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [0, 0, 0],
    [255, 255, 255]
]

def lru_cache_with_hash(hash_function):
    def decorator(func):
        cache = {}
        def wrapped_func(self, arg):
            hash_key = hash_function(arg)
            if hash_key in cache:
                return cache[hash_key]
            result = func(self, arg)
            cache[hash_key] = result
            return result
        return wrapped_func
    return decorator

def topk (arr, k) :
    """
    Find top k indices

    Parameters:
    arr (np.ndarray): A 1D NumPy array.

    Returns:
    np.ndarray: k indices
    """
    return np.argsort(arr)[-k:]

def avg (lst) : 
    """ 
    use this instead of np.mean if you are worried
    about empty lists. if list is actually an iterator,
    this function will consume the iterator.
    """
    lst = list(lst)
    if len(lst) == 0 : 
        return 0
    return sum(lst) / len(lst)

def hashPILImage(img):
   return hashlib.md5(img.tobytes()).hexdigest()

def filter_predicted_polygons (polygon_preds, confidence_scores, top_k=40, cluster_size_threshold=0.05) : 
    top_idx = topk(confidence_scores, top_k) 
    polygon_preds = np.array([polygon_preds[i] for i in top_idx])
    X = polygon_preds.reshape(top_k, -1)
    clustering = MeanShift(cluster_all=False).fit(X)
    bins = [[] for _ in range(max(clustering.labels_) + 1)] 
    for i in range(top_k): 
        bins[clustering.labels_[i]].append(X[i].reshape(-1, 2))
    cluster_size = int(cluster_size_threshold * top_k)
    bins = [b for b in bins if len(b) >= cluster_size]
    polygons = [avg(b).astype(int) for b in bins]
    return polygons

def parse_ckpt_path(s):
    pattern = r".*epoch=(\d+)-step=(\d+).ckpt"
    match = re.search(pattern, s)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return None

def load_model (expt_log_dir, extra_args=dict()) : 
    ckpt_dir = osp.join(expt_log_dir, 'checkpoints')
    ckpt_paths = listdir(ckpt_dir)
    if any('last.ckpt' in _ for _ in ckpt_paths) : 
        # check for last.ckpt
        ckpt_path = [_ for _ in ckpt_paths if 'last.ckpt' in _][0]
    else  :
        # find the checkpoint with maximum steps 
        valid_paths = [_ for _ in ckpt_paths if parse_ckpt_path(_) is not None]
        ckpt_path = sorted(valid_paths, key=lambda x: parse_ckpt_path(x)[1])[-1]

    print('Loading from checkpoint ...', ckpt_path)

    # load model args
    args_dict = osp.join(expt_log_dir, 'args.pkl') 
    with open(args_dict, 'rb') as fp :
        dict_ = pickle.load(fp)
        dict_.update(extra_args)
        args = DictWrapper(dict_)

    # make model
    model = ComicFramePredictorModule.load_from_checkpoint(ckpt_path, args=args)
    return model

class ComicFramePredictorModule(pl.LightningModule):

    def __init__(self, args):
        super(ComicFramePredictorModule, self).__init__()
        self.sam_model = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_path)
        self.projector_x = MLP(2 * 256, 256, 4, 3).float()
        self.projector_y = MLP(2 * 256, 256, 4, 3).float()

        self.point_confidence_score_predictor = MLP(2 * 256, 256, 1, 3).float()

        self.args = args

    def forward(self, batch, stage='train'):
        prefix = '' if stage == 'train' else 'val_'
        if 'features' in batch : 
            features = batch['features']
        else : 
            assert 'img' in batch, "Either image or features needed for this" 
            with torch.no_grad() : 
                features = self.sam_model.image_encoder(batch['img'])
        point_coords = batch['point_coords']
        point_labels = batch['point_labels']
        original_size = batch['original_size']
        input_size = batch['input_size']
        shape = batch['shape']
        index = batch['index']
        point_confidence_score = batch['point_confidence_score'] 
        N = features.shape[0]

        points = (point_coords, point_labels)

        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # Predict masks
        low_res_masks, iou_predictions, prompt_tokens = self.sam_model.mask_decoder(
            image_embeddings=features,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            interleave=False, # this ensures correct behaviour when each prompt is for a different image
            return_prompt_tokens=True
        )

        out_x = self.projector_x(prompt_tokens.reshape(N, -1)) # [N, 4]
        out_y = self.projector_y(prompt_tokens.reshape(N, -1)) # [N, 4]

        point_confidence_score_pred = self.point_confidence_score_predictor(prompt_tokens.reshape(N, -1)) # [N, 1]

        out = torch.cat((out_x[..., None], out_y[..., None]), 2) # [N, 4, 2]

        l1 = F.l1_loss(out, shape)
        l2 = F.mse_loss(out, shape)

        angle_loss = rel_orientation_loss(out, shape)

        point_confidence_l1 = F.l1_loss(point_confidence_score_pred, point_confidence_score)

        loss = 0.5 * l1 + 0.25 * l2 + 0.25 * angle_loss + 0.05 * point_confidence_l1

        return {
            'pred': out,
            f'{prefix}loss': loss,
            f'{prefix}loss_l1': l1,
            f'{prefix}loss_l2': l2,
            f'{prefix}loss_angle': angle_loss,
            f'{prefix}loss_point_confidence': point_confidence_l1, 
            'low_res_masks': low_res_masks,
            'iou_predictions': iou_predictions,
        }

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        for k in outputs.keys() : 
            if 'loss' in k: 
                self.log(k, outputs[k])
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self(batch, stage='val')
        for k in outputs.keys() : 
            if 'loss' in k: 
                self.log(k, outputs[k])
        return outputs

    @torch.no_grad()
    def run_inference_full (self, np_img, num_sample=20, top_k=40, cluster_size_threshold=0.05) :
        """
        Creates a 2D grid of query points on the image and queries these 
        """
        # Import some stuff we need ...
        from datamodule import original_size_to_input_size, normalized_point_to_image_point

        # Create a 2D grid of points
        Y, X = np_img.shape[:2]
        max_sz = max(Y, X)
        XS = np.linspace(0, X, num_sample)
        YS = np.linspace(0, Y, num_sample)
        XS, YS = np.meshgrid(XS, YS)
        points_grid = np.concatenate((XS.reshape(-1, 1), YS.reshape(-1, 1)), axis=1).astype(int)

        # precompute and cache sam features so that we don't go crazy
        img_cpy = deepcopy(np_img)
        img = apply_transform_to_pil_without_sam_model(Image.fromarray(img_cpy), 'cpu').cuda()
        features = self.sam_model.image_encoder(img)

        # Prepare data ...
        transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)
        original_size = img_cpy.shape[:2]
        input_size = original_size_to_input_size(transform, original_size)

        polygon_preds = []
        confidence_scores = []

        point_labels = torch.ones((1, 1)).float().cuda()
        for point in points_grid :
            points = torch.from_numpy(transform.apply_coords(point.reshape(1, -1).astype(np.float32), original_size))[None, ...].cuda()
            points = (points, point_labels)

            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )

            # Predict masks ... 
            low_res_masks, iou_predictions, prompt_tokens = self.sam_model.mask_decoder(
                image_embeddings=features,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                interleave=False, # this ensures correct behaviour when each prompt is for a different image
                return_prompt_tokens=True
            )

            out_x = self.projector_x(prompt_tokens.reshape(1, -1)) # [N, 4]
            out_y = self.projector_y(prompt_tokens.reshape(1, -1)) # [N, 4]

            point_confidence_score_pred = self.point_confidence_score_predictor(prompt_tokens.reshape(1, -1)) # [N, 1]

            out = torch.cat((out_x[..., None], out_y[..., None]), 2) # [N, 4, 2]
            out = out.squeeze()
            pts = normalized_point_to_image_point(out, input_size, original_size).detach().cpu().numpy().astype(int)

            polygon_preds.append(pts)
            confidence_scores.append(point_confidence_score_pred.squeeze().item())

        polygons = filter_predicted_polygons(polygon_preds, confidence_scores, top_k, cluster_size_threshold)

        for i, pts in enumerate(polygons) : 
            for point in pts:
                cv2.drawMarker(img_cpy, point, COLORS[i % len(COLORS)], markerType=5, markerSize=int(0.05 * max_sz), thickness=int(0.01 * max_sz))

        return aspectRatioPreservingResize(img_cpy, 256)

    @torch.no_grad()
    @log_to_dir()
    def run_inference (self, np_img, point) : 
        """ 
        np_img - [H, W, 3]
        point  - [(x, y), ...], ideally just 1 point
        """
        # Import some stuff we need ...
        from datamodule import original_size_to_input_size, normalized_point_to_image_point

        img_cpy = deepcopy(np_img)

        # Prepare data ...
        transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

        original_size = img_cpy.shape[:2]
        input_size = original_size_to_input_size(transform, original_size)

        img = apply_transform_to_pil_without_sam_model(Image.fromarray(img_cpy), 'cpu').cuda()

        point = torch.from_numpy(transform.apply_coords(np.array(point).astype(np.float32), original_size))[None, ...].cuda() # [1, 1, 2]
        point_labels = torch.ones((1, 1)).float().cuda()

        # Do inference ... 
        features = self.sam_model.image_encoder(img)

        points = (point, point_labels)

        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # Predict masks ... 
        low_res_masks, iou_predictions, prompt_tokens = self.sam_model.mask_decoder(
            image_embeddings=features,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            interleave=False, # this ensures correct behaviour when each prompt is for a different image
            return_prompt_tokens=True
        )

        out_x = self.projector_x(prompt_tokens.reshape(1, -1)) # [N, 4]
        out_y = self.projector_y(prompt_tokens.reshape(1, -1)) # [N, 4]

        point_confidence_score_pred = self.point_confidence_score_predictor(prompt_tokens.reshape(1, -1)) # [N, 1]

        out = torch.cat((out_x[..., None], out_y[..., None]), 2) # [N, 4, 2]
        out = out.squeeze()
        pts = normalized_point_to_image_point(out, input_size, original_size).detach().cpu().numpy().astype(int)

        # Make visualization
        for point in pts:
            cv2.drawMarker(img_cpy, point, [255, 0, 0], markerType=5, markerSize=20, thickness=5)

        return img_cpy, pts.tolist()

    @torch.no_grad()
    def encode_image (self, np_img) : 
        """ 
        Use the heavy weight encoder for this 
        """
        from datamodule import original_size_to_input_size, normalized_point_to_image_point

        img_cpy = deepcopy(np_img)

        # Prepare data ...
        transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

        original_size = img_cpy.shape[:2]
        input_size = original_size_to_input_size(transform, original_size)

        img = apply_transform_to_pil_without_sam_model(Image.fromarray(img_cpy), 'cpu').cuda()

        # Do inference ... 
        features = self.sam_model.image_encoder(img)
        return features

    @lru_cache_with_hash(hashPILImage)
    @torch.no_grad()
    def encode_image_pil (self, pil_img) : 
        """ 
        Use the heavy weight encoder for this 
        """
        from datamodule import original_size_to_input_size, normalized_point_to_image_point

        # Prepare data ...
        transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

        width, height = pil_img.size 
        original_size = (height, width)
        input_size = original_size_to_input_size(transform, original_size)

        img = apply_transform_to_pil_without_sam_model(pil_img, 'cpu').cuda()

        # Do inference ... 
        features = self.sam_model.image_encoder(img)
        return features

    @torch.no_grad()
    def run_inference_simple (self, np_img, point, features=None) : 
        """ 
        np_img - [H, W, 3]
        point  - [(x, y), ...], ideally just 1 point
        
        No logging and no visualization

        """
        # Import some stuff we need ...
        from datamodule import original_size_to_input_size, normalized_point_to_image_point

        img_cpy = deepcopy(np_img)

        # Prepare data ...
        transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

        original_size = img_cpy.shape[:2]
        input_size = original_size_to_input_size(transform, original_size)

        img = apply_transform_to_pil_without_sam_model(Image.fromarray(img_cpy), 'cpu').cuda()

        point = torch.from_numpy(transform.apply_coords(np.array(point).astype(np.float32), original_size))[None, ...].cuda() # [1, 1, 2]
        point_labels = torch.ones((1, 1)).float().cuda()

        # Do inference ... 
        if features is None :
            features = self.sam_model.image_encoder(img)

        points = (point, point_labels)

        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # Predict masks ... 
        low_res_masks, iou_predictions, prompt_tokens = self.sam_model.mask_decoder(
            image_embeddings=features,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            interleave=False, # this ensures correct behaviour when each prompt is for a different image
            return_prompt_tokens=True
        )

        out_x = self.projector_x(prompt_tokens.reshape(1, -1)) # [N, 4]
        out_y = self.projector_y(prompt_tokens.reshape(1, -1)) # [N, 4]

        point_confidence_score_pred = self.point_confidence_score_predictor(prompt_tokens.reshape(1, -1)) # [N, 1]

        out = torch.cat((out_x[..., None], out_y[..., None]), 2) # [N, 4, 2]
        out = out.squeeze()
        pts = normalized_point_to_image_point(out, input_size, original_size).detach().cpu().numpy().astype(int)

        return pts.tolist()
    
    @torch.no_grad()
    def run_inference_simple_pil (self, pil_img, point, features=None) : 
        """ 
        pil_img - 3 channel PIL Image
        point  - [(x, y), ...], ideally just 1 point
        
        No logging and no visualization

        """
        # Import some stuff we need ...
        from datamodule import original_size_to_input_size, normalized_point_to_image_point

        # Prepare data ...
        transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

        width, height = pil_img.size
        original_size = (height, width)
        input_size = original_size_to_input_size(transform, original_size)

        point = torch.from_numpy(transform.apply_coords(np.array(point).astype(np.float32), original_size))[None, ...].cuda() # [1, 1, 2]
        point_labels = torch.ones((1, 1)).float().cuda()

        # Do inference ... 
        if features is None :
            features = self.encode_image_pil(pil_img) 

        points = (point, point_labels)

        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # Predict masks ... 
        low_res_masks, iou_predictions, prompt_tokens = self.sam_model.mask_decoder(
            image_embeddings=features,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            interleave=False, # this ensures correct behaviour when each prompt is for a different image
            return_prompt_tokens=True
        )

        out_x = self.projector_x(prompt_tokens.reshape(1, -1)) # [N, 4]
        out_y = self.projector_y(prompt_tokens.reshape(1, -1)) # [N, 4]

        point_confidence_score_pred = self.point_confidence_score_predictor(prompt_tokens.reshape(1, -1)) # [N, 1]

        out = torch.cat((out_x[..., None], out_y[..., None]), 2) # [N, 4, 2]
        out = out.squeeze()
        pts = normalized_point_to_image_point(out, input_size, original_size).detach().cpu().numpy().astype(int)

        return pts.tolist()

    def configure_optimizers(self):
        return torch.optim.Adam(list(self.projector_x.parameters()) \
                              + list(self.projector_y.parameters()) \
                              + list(self.sam_model.mask_decoder.parameters()) \
                              + list(self.sam_model.prompt_encoder.parameters()),
            lr=self.args.lr
        )

if __name__ == "__main__" :
    # standalone model test
    from args import get_parser
    from datamodule import FrameDataModule

    parser = get_parser()
    args = parser.parse_args()

    data_module = FrameDataModule(args)
    model = ComicFramePredictorModule(args)

    data_module.setup() 
    for batch in data_module.train_dataloader() : 
        break

    model.training_step(batch, 0)


