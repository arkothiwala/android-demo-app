# Imports
import os, torch, torchvision, fastai, shutil, glob, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from fastai.vision import *
from fastai.metrics import error_rate



def _crop_pad_default(x, size, padding_mode='reflection', row_pct:uniform = 0.5, col_pct:uniform = 0.5):
    "Crop and pad tfm - `row_pct`,`col_pct` sets focal point."
    padding_mode = _pad_mode_convert[padding_mode]
    size = tis2hw(size)
    if x.shape[1:] == torch.Size(size): return x
    rows,cols = size
    row_pct,col_pct = _minus_epsilon(row_pct,col_pct)
    if x.size(1)<rows or x.size(2)<cols:
        row_pad = max((rows-x.size(1)+1)//2, 0)
        col_pad = max((cols-x.size(2)+1)//2, 0)
        x = F.pad(x[None], (col_pad,col_pad,row_pad,row_pad), mode=padding_mode)[0]
    row = int((x.size(1)-rows+1)*row_pct)
    col = int((x.size(2)-cols+1)*col_pct)
    x = x[:, row:row+rows, col:col+cols]
    return x.contiguous() # without this, get NaN later - don't know why

#mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250])
def _normalize_batch(b:Tuple[Tensor,Tensor], mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x,y = b
    mean,std = mean.to(x.device),std.to(x.device)
    if do_x: x = normalize(x,mean,std)
    if do_y and len(y.shape) == 4: y = normalize(y,mean,std)
    return x,y

def get_resize_target(img, crop_target, do_crop=False)->TensorImageSize:
    "Calc size of `img` to fit in `crop_target` - adjust based on `do_crop`."
    if crop_target is None: return None
    ch,r,c = img.shape
    target_r,target_c = crop_target
    ratio = (min if do_crop else max)(r/target_r, c/target_c)
    return ch,int(round(r/ratio)),int(round(c/ratio)) #Sometimes those are numpy numbers and round doesn't return an int.

if __name__ == "__main__":
    results = []
    np.random.seed(47)
    data_path = 'data'
    data = ImageDataBunch.from_folder(data_path, train = 'train', test= 'test', valid_pct=0.15, ds_tfms=get_transforms(), size=224, bs=16
                                    ).normalize(imagenet_stats)
    for i in tqdm(range(len(data.test_ds))):
        im = Image(data.test_ds.x[0].data)
        resize_target = get_resize_target(im, (224, 224), do_crop=True)
        print(resize_target)
        im.resize(resize_target) # this does affine_transform
        im_crop_pad = _crop_pad_default(im.data, (224,224))
        im_normalized = _normalize_batch((im_crop_pad,None),\
            mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))[0]