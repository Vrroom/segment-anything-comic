import pytorch_lightning as pl
import torch
from torch import nn
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.mask_decoder import MLP
import torch.nn.functional as F
from itertools import chain

class ComicFramePredictorModule(pl.LightningModule):

    def __init__(self, args):
        super(ComicFramePredictorModule, self).__init__()
        self.sam_model = sam_model_registry["vit_h"](checkpoint="./checkpoints/sam_vit_h_4b8939.pth")
        self.projector_x = MLP(2 * 256, 256, 4, 3).float()
        self.projector_y = MLP(2 * 256, 256, 4, 3).float()
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

        out = torch.cat((out_x[..., None], out_y[..., None]), 2) # [N, 4, 2]

        l1 = F.l1_loss(out, shape)
        l2 = F.mse_loss(out, shape)

        loss = 0.5 * l1 + 0.5 * l2

        return {
            'pred': out,
            f'{prefix}loss': loss,
            f'{prefix}loss_l1': l1,
            f'{prefix}loss_l2': l2,
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


