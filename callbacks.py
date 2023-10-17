from pytorch_lightning import Callback
from osTools import *
from datamodule import * 

class VisualizePoints(Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        if global_step % 100 == 0 :
            save_path_base = osp.join(pl_module.logger.log_dir, 'images', 'train')
            mkdir(save_path_base)
            save_to = osp.join(save_path_base, f'img_{global_step}.png')
            visualize_batch(pl_module.sam_model, batch, trainer.datamodule.train_data, outputs=outputs, save_to=save_to)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        save_path_base = osp.join(pl_module.logger.log_dir, 'images', 'val')
        mkdir(save_path_base)
        save_to = osp.join(save_path_base, f'img_{global_step}.png')
        visualize_batch(pl_module.sam_model, batch, trainer.datamodule.test_data, outputs=outputs, save_to=save_to)
