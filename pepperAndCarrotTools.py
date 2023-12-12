from osTools import * 
from tqdm import tqdm
from PIL import Image
import json
import random

def pepper_and_carrot_generator (dataset_path, split='val') : 
    assert split in ['val', 'test'], "Split has to be either \'val\' or \'test\'"
    rng = random.Random(0)
    json_file = osp.join(dataset_path, 'annotations.json') 
    with open(json_file) as fp : 
        annotations = json.load(fp) 

    idx = list(range(len(annotations)))
    rng.shuffle(idx)
    N = int(len(annotations) * 0.1)
    st, en = 0 if split == 'val' else N, N if split == 'val' else len(annotations)
    idx = idx[st:en]

    annotations = [annotations[i] for i in idx]
    image_paths = [osp.join(dataset_path, annot['data']['image']) for annot in annotations]

    for i, (img_path, annot) in enumerate(zip(image_paths, annotations)) : 
        img = Image.open(img_path)
        original_size = tuple(reversed(img.size))
        Y, X = original_size
        shapes = []
        for result in annot['annotations'][0]['result']: 
            points = result['value']['points']
            shape = [(int(X * a / 100), int(Y * b / 100)) for a, b in points]
            shapes.append(shape)
        yield dict(img=img, original_size=original_size, shapes=shapes)

if __name__ == "__main__" : 
    generator = pepper_and_carrot_generator('../pepper_and_carrot_imgs/', split='test')
    for item in tqdm(generator) :
        pass


