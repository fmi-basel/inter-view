import numpy as np
import PIL.Image

from bokeh.palettes import viridis, d3, brewer, inferno
from bokeh.transform import factor_cmap, linear_cmap


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


def update_color_mapping(source, hue):
    ''' Determines if 'hue' refer to categorical or numerical data and sets the 'color' column in 'source' accordingly.
        
        NOTE:
        ----
        
        color mapping with bokeh cmap on client side is buggy --> pre-compute color in data source column
        '''

    if hue == 'disabled':
        source.data['color'] = ['grey' for c in source.data['color']]
        return {'hue_type': 'disabled', 'cmap': None}

    else:

        if any(isinstance(val, str)
               for val in source.data[hue]):  # categorical column
            factors = sorted(list(set(str(f) for f in source.data[hue])))
            palette_length = min(256, len(factors))

            if palette_length <= 2:
                palette = brewer['Set1'][3][0:palette_length]
            elif palette_length <= 9:
                palette = brewer['Set1'][palette_length]
            elif palette_length <= 20:
                palette = d3['Category20'][palette_length]
            else:
                palette = viridis(palette_length)

            lut = dict(zip(factors, palette))
            source.data['color'] = [lut[f] for f in source.data[hue]]

            return {
                'hue_type':
                'category',
                'cmap':
                factor_cmap(field_name=hue, palette=palette, factors=factors)
            }

        else:  # numerical column
            palette = viridis(256)
            low = min(source.data[hue])
            high = max(source.data[hue])

            source.data['color'] = [
                palette[int((v - low) / (high - low) * 255)]
                for v in source.data[hue]
            ]

            # return colormap to update colorbars
            return {
                'hue_type':
                'numeric',
                'cmap':
                linear_cmap(field_name=hue,
                            palette=viridis(256),
                            low=low,
                            high=high)
            }


def read_image_size(path):
    img = PIL.Image.open(path)
    return img.size
