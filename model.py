import pytorch_lightning as pl
import torch
from torch import nn
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.mask_decoder import MLP
import torch.nn.functional as F

class ComicFramePredictorModule(pl.LightningModule):

    def __init__(self, args):
        super(ComicFramePredictorModule, self).__init__()
        self.sam_model = sam_model_registry["vit_h"](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
        self.projector = MLP(2 * 256, 256, 8, 3).float()
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
        N = features.shape[0]

        points = (point_coords, point_labels)

        with torch.no_grad(): 
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

        out = self.projector(prompt_tokens.reshape(N, -1)).reshape(N, 4, 2)
        loss = F.mse_loss(out, shape)
        return {
            'pred': out,
            f'{prefix}loss': loss,
            'low_res_masks': low_res_masks,
            'iou_predictions': iou_predictions,
        }

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        self.log('train_loss', outputs['loss'])
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self(batch, stage='val')
        self.log('val_loss', outputs['val_loss'])
        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.projector.parameters(), lr=self.args.lr)

