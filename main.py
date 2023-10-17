import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from datamodule import FrameDataModule
from model import ComicFramePredictorModule
from args import get_parser
import torch
from callbacks import *

def main():
    torch.set_float32_matmul_precision('medium')

    parser = get_parser()
    args = parser.parse_args()

    seed_everything(args.seed)

    data_module = FrameDataModule(args)
    model = ComicFramePredictorModule(args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator='gpu',
        callbacks=[VisualizePoints()],
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()

