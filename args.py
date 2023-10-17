import argparse

class DictWrapper:
    def __init__(self, d):
        self._dict = d

    def __getattr__(self, name):
        if name in self._dict:
            return self._dict[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        if name == '_dict':
            super().__setattr__(name, value)
        else:
            self._dict[name] = value

    def __delattr__(self, name):
        if name in self._dict:
            del self._dict[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

def get_parser () : 
    parser = argparse.ArgumentParser(description='Comic SAM')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--sam_ckpt_path', type=str, default='./checkpoints/sam_vit_h_4b8939.pth', help='path to sam checkpoint')
    parser.add_argument('--base_dir', type=str, default='../comic_data', help='path to dataset')
    return parser

