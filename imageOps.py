from PIL import Image
import numpy as np
import base64, io

ENCODING = 'utf-8'

def make_image_grid (images, row_major=True, gutter=True):
    """
    Make a large image where the images are stacked.

    images: list/list of list of PIL Images
    row_major: if images is list, whether to lay them down horizontally/vertically
    """
    assert isinstance(images, list) and len(images) > 0, "images is either not a list or an empty list"
    if isinstance(images[0], list) :
        return make_image_grid([make_image_grid(row) for row in images], False)
    else :
        if row_major :
            H = min(a.size[1] for a in images)
            images = [a.resize((int(H * a.size[0] / a.size[1]), H)) for a in images]
            W = sum(a.size[0] for a in images)
            gutter_width = int(0.01 * W) if gutter else 0
            W += (len(images) - 1) * gutter_width
            img = Image.new('RGB', (W, H))
            cSum = 0
            for a in images :
                img.paste(a, (cSum, 0))
                cSum += (a.size[0] + gutter_width)
        else :
            W = min(a.size[0] for a in images)
            images = [a.resize((W, int(W * a.size[1] / a.size[0]))) for a in images]
            H = sum(a.size[1] for a in images)
            gutter_width = int(0.01 * W) if gutter else 0
            H += (len(images) - 1) * gutter_width
            img = Image.new('RGB', (W, H))
            cSum = 0
            for a in images :
                img.paste(a, (0, cSum))
                cSum += (a.size[1] + gutter_width)
        return img

def imsave (arr, fname) : 
    """ utility for saving numpy array """ 
    imgToPIL(arr).save(fname)

def imgArrayToPIL (arr) :
    """ utility to convert img array to PIL """
    if arr.dtype in [np.float32, np.float64, float] :
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype in [np.int32, np.int64, int]:
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType)

def aspectRatioPreservingResize (arr, smaller_dim) :
    """ utility for resizing image, ensuring that smaller dimension matches """
    pil_img = imgArrayToPIL(arr)
    h, w = pil_img.size
    if h < w :
        h, w = smaller_dim, smaller_dim * w / h
    else :
        h, w = smaller_dim * h / w, smaller_dim
    h, w = int(h), int(w)
    resized = pil_img.resize((h, w))
    np_arr = np.array(resized).astype(arr.dtype)
    if arr.dtype in [float, np.float32, np.float64] :
        np_arr /= 255.0
    return np_arr

def aspectRatioPreservingResizePIL (pil_img, smaller_dim) :
    """ utility for resizing image, ensuring that smaller dimension matches """
    h, w = pil_img.size
    if h < w :
        h, w = smaller_dim, smaller_dim * w / h
    else :
        h, w = smaller_dim * h / w, smaller_dim
    h, w = int(h), int(w)
    resized = pil_img.resize((h, w))
    return resized
    
def imgArrayToPILRGB (arr) : 
    """ utility to convert img array to 3 channel PIL """
    from vectorrvnn.utils import alphaComposite
    arr = alphaComposite(arr)
    if arr.dtype in [np.float32, np.float64] : 
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype == np.int: 
        arr = arr.astype(np.uint8)
    assert(arr.dtype == np.uint8)
    chanType = "RGBA" if arr.shape[2] == 4 else "RGB"
    return Image.fromarray(arr, chanType).convert("RGB")

def bytes2PIL(b64Str):
    imgdata = base64.b64decode(str(b64Str))
    return Image.open(io.BytesIO(imgdata))

def PIL2byteStr(img):
    img = np.array(img)
    buffered = io.BytesIO()
    image = Image.fromarray(img)
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode(ENCODING)

