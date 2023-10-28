import torch
from copy import deepcopy
import torch.nn as nn
from more_itertools import flatten
from functools import partial, lru_cache
import numpy as np
from torchvision.models import *
from torchvision.ops import *

def clones(module, N) :
    copies = [deepcopy(module) for _ in range(N)]
    return nn.ModuleList(copies)

def tensorApply (thing, fn, 
    predicate=lambda x: True, module=torch) : 
    """ 
    Apply a tensor transformation to a tensor 
    or dictionary or a list.
    """
    Cls = torch.Tensor if module == torch else np.ndarray
    if isinstance(thing, Cls) and predicate(thing): 
        thing = fn(thing)
    elif isinstance(thing, dict) : 
        for k, v in thing.items() : 
            thing[k] = tensorApply(v, fn, predicate, module)
    elif isinstance(thing, list) : 
        for i, _ in enumerate(thing) : 
            thing[i] = tensorApply(thing[i], fn, 
                    predicate, module)
    return thing

def tensorFilter(thing, predicate, module=torch) :
    """
    Filter tensors from list/dictionary or
    just an individual collection based on whether 
    it passes a predicate. Returns a list.
    """
    Cls = torch.Tensor if module == torch else np.ndarray
    if isinstance(thing, Cls) and predicate(thing): 
        return [thing]
    elif isinstance(thing, dict) : 
        return list(flatten(map(
            partial(
                tensorFilter, 
                predicate=predicate,
                module=module
            ), 
            map(
                lambda k : thing[k],
                sorted(thing.keys())
            )
        )))
    elif isinstance(thing, list) : 
        return list(flatten(map(
            partial(
                tensorFilter, 
                predicate=predicate,
                module=module
            ), 
            thing
        )))
    return []

def setParameterRequiresGrad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def l2 (a, b, eps=1e-5) : 
    """ 
    a.shape == b.shape == [B, N].
    B is the batch size and N is the dimension
    of the embedding.
    """
    d2 = torch.sum((a - b) ** 2, dim=1, keepdims=True)
    l2 = torch.sqrt(d2 + eps)
    return l2

def unitNorm(a, dim=1) : 
    return a / a.norm(dim=dim, keepdim=True)

def ncs (a, b) : 
    """
    a.shape == b.shape == [B, N].
    B is the batch size and N is the dimension
    of the embedding.
    """
    return - (unitNorm(a) * unitNorm(b)).sum(dim=1, keepdim=True)

def maskedMean(thing, mask, eps=1e-5) : 
    """
    thing.shape == mask.shape == [B, 1]
    Used to handle cases where nothing is 
    in the thing after mask is applied
    """
    return thing[mask].sum() / (mask.sum() + eps)

def normalize2UnitRange (thing) :
    m, M = thing.min(), thing.max()
    if m == M : 
        return torch.ones_like(thing)
    else : 
        return (thing - m) / (M - m)

def channelDim (thing, module=torch) : 
    cId = 0 if module == torch else 2
    if len(thing.shape) == 4 : 
        return cId + 1
    elif len(thing.shape) == 3 : 
        return cId
    else : 
        return None

def channels (thing, module=torch) : 
    idx = channelDim(thing, module)
    if idx is not None:  
        return thing.shape[idx]
    return None

def isImage (thing, module=torch) : 
    return (len(thing.shape) in [3, 4]) \
            and (channels(thing) in [1, 3, 4])

def isGreyScale (thing, module=torch) : 
    return isImage(thing, module) \
            and channels(thing) == 1

def toGreyScale (im, module=torch) : 
    return torch.cat((im, im, im), channelDim(im, module))

def freezeLayers (model, freeze_layers) :
    for name, module in model.named_modules() : 
        if name in freeze_layers : 
            module.requires_grad_(False)

def addLayerNorm (model, output_size) : 
    return nn.Sequential(
        model, 
        nn.LayerNorm(output_size)
    )

@lru_cache(maxsize=128)
def sharedConvBackbone (opts): 
    return convBackbone(opts)

def convBackbone (opts) : 
    print('Initializing', opts.backbone)
    backboneFn = globals()[opts.backbone]
    model = backboneFn(pretrained=True)
    if opts.backbone.startswith('resnet') : 
        inFeatures = model.fc.in_features
        model.fc = nn.Linear(inFeatures, opts.embedding_size, bias=not opts.use_layer_norm)
        model.fc.apply(getInitializer(opts))
        if opts.use_layer_norm : 
            model.fc = addLayerNorm(model.fc, opts.embedding_size)
    elif opts.backbone in ['alexnet', 'vgg16', 'vgg16_bn'] : 
        inFeatures = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(inFeatures, opts.embedding_size, bias=not opts.use_layer_norm)
        model.classifier[-1].apply(getInitializer(opts))
        if opts.use_layer_norm : 
            model.classifier[-1] = addLayerNorm(model.classifier[-1], opts.embedding_size)
    else : 
        raise ValueError(f'{opts.backbone} not supported')
    freezeLayers(model, opts.freeze_layers)
    model = model.float()
    return model

def fcn (opts, input_size, output_size) :
    """
    Make a fully connected model with given input and output size

    The intermediate layers are provided by the hidden_size parameter
    in opts. Each linear layer is followed by a ReLU non linearity with 
    the exception of the final layer. All layers are then initialized 
    by the initialization scheme in opts.
    """
    hidden_size = opts.hidden_size
    repr = f'{input_size} {" ".join(map(str, hidden_size))} {output_size}'
    print(f'Initializing FCN({repr})')
    sizes = [input_size, *hidden_size, output_size]
    inOuts = list(zip(sizes, sizes[1:]))
    topLayers = [
        nn.Sequential(
            nn.Linear(*io),
            nn.ReLU(),
            nn.Dropout(p=opts.dropout)
        )
        for io in inOuts[:-1]
    ]
    lastLayer = nn.Linear(*inOuts[-1], bias=not opts.use_layer_norm)
    model = nn.Sequential(*topLayers, lastLayer)
    if opts.use_layer_norm : 
        model = addLayerNorm(model, output_size)
    model.apply(getInitializer(opts))
    return model

def _computeGradient (model, input_shape, index) : 
    """ 
    Compute the gradient of output neuron at given
    index from an input of all ones.
    """
    input = torch.ones(input_shape)
    input.requires_grad = True
    output = model(input)
    output[tuple(index)].backward()
    return input.grad * input.grad

def _dummymodel (model) : 
    """ 
    Initialize a copy of the model with all weights being 1
    """
    model_ = deepcopy(model)
    for p in model_.parameters() : 
        nn.init.constant_(p.data, 1)
    return model_

def receptiveField (model, input_shape, index) :
    """ 
    Find the receptive field or the number of input neurons
    that effect a particular neuron at output by backpropagation.
    """
    model_ = _dummymodel(model)
    input_shape, index = [1, *input_shape], [0, *index]
    grad = _computeGradient(model_, input_shape, index)
    effectedElts = (grad > 0).sum().item()
    return effectedElts

def cnnReceptiveField (cnn, input_shape, index) : 
    """
    Input shape is assumed to be (C, ...). Where
    C is the number of channels.
    """
    C, *_ = input_shape
    dims = len(input_shape) - 1
    effectedElts = receptiveField(cnn, input_shape, index)
    return int((effectedElts / C) ** (1. / dims))

def cnnEffectiveStride (cnn, input_shape, index) : 
    input_shape, index = [1, *input_shape], [0, *index]
    cnn_ = _dummymodel(cnn)
    strides = []
    idxDims = list(range(len(index)))
    for i in idxDims[2:] : 
        # Compute grad map for index 
        grad1 = _computeGradient(cnn_, input_shape, index)
        # Compute grad map for index incremented along stride direction
        index_ = deepcopy(index)
        index_[i] += 1
        grad2 = _computeGradient(cnn_, input_shape, index_)
        # The affected inputs form a hypercube (a square for 2D CNNs). 
        # We want to find those inputs that effect only one of the output 
        # neurons i.e. those at index and index_ but not both.
        symDiff = ((grad1 > 0) ^ (grad2 > 0)).sum()
        # Now find the size of the hypercube obtained by removing the 
        # current stride direction. 
        hyperplaneDims = [_ for _ in idxDims if _ != i]
        hyperplane = (grad1 > 0).sum(hyperplaneDims).max()
        stride = (symDiff / (2 * hyperplane)).item()
        strides.append(stride)
    return strides

def toNumpyCPU(tens) :
    return tens.detach().cpu().numpy()

def lte (a, b) : 
    """
    Find number of instances when some
    element in a is <= some element in b
    """
    a = a.view((1, -1))
    b = b.view((-1, 1))
    a = torch.repeat_interleave(a, b.size(0), 0)
    return a <= b

def clipGradients (model, max_grad_norm=None) : 
    if max_grad_norm is not None:
        nrm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

def toTorchImage (im) : 
    """ convert a raster into a torch image for visdom """
    im = torch.from_numpy(im).permute((2, 0, 1))
    im = alphaComposite(im, module=torch)
    return im

""" obtained this from https://discuss.pytorch.org/t/batched-index-select/9115/8 """
def batched_index_select(input, dim, index):
	views = [input.shape[0]] + \
			[1 if i != dim else -1 for i in range(1, len(input.shape))]
	expanse = list(input.shape)
	expanse[0] = -1
	expanse[dim] = -1
	index = index.view(views).expand(expanse)
	return torch.gather(input, dim, index)

