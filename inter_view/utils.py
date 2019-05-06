import numpy as np


def min_max_scaling(img,
                    min_val=0,
                    max_val=1,
                    eps=1e-5,
                    saturation=0.0001,
                    separate_channels=True):
    '''Re-scale input to be within [min_val, max_val]
    
    '''
    if separate_channels:
        axis = tuple(range(img.ndim - 1))
        keepdims = True
    else:
        axis = None
        keepdims = False

    img = img.astype(dtype=np.float, copy=True)
    img -= np.quantile(img, saturation, axis=axis, keepdims=keepdims)  # min
    img = img / (np.quantile(img, 1 - saturation, axis=axis, keepdims=keepdims)
                 + eps) * (max_val - min_val)
    img = img + min_val
    img = np.clip(img, min_val, max_val)

    return img
