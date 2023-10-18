import pytorch_lightning as pl
import logging
from pytorch_lightning import seed_everything
from datamodule import FrameDataModule
from model import ComicFramePredictorModule
from args import get_parser
import torch
from callbacks import *

def print_dict_as_table(d):
    # Determine the maximum width of the keys and values for alignment
    max_key_width = max(len(str(key)) for key in d.keys())
    max_value_width = max(len(str(value)) for value in d.values())

    # Print the table header
    print(f"{'Key':<{max_key_width}} | {'Value':<{max_value_width}}")
    print('-' * (max_key_width + max_value_width + 3))

    # Print the key-value pairs
    for key, value in d.items():
        print(f"{str(key):<{max_key_width}} | {str(value):<{max_value_width}}")

def main():
    torch.set_float32_matmul_precision('medium')

    parser = get_parser()
    args = parser.parse_args()

    print_dict_as_table(vars(args))

    seed_everything(args.seed)

    data_module = FrameDataModule(args)
    model = ComicFramePredictorModule(args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.gpus,
        accelerator='gpu',
        callbacks=[VisualizePoints()],
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()

