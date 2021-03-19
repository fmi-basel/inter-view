import numpy as np
import holoviews as hv
hv.extension('bokeh', logo=False)
import param
import panel as pn
import matplotlib.pyplot as plt

from holoviews.operation.datashader import rasterize
from bokeh.models import WheelZoomTool
from holoviews.core import Store

valid_rgb_options = [
    k for group in ['style', 'plot', 'norm', 'output']
    for k in Store.options(backend='bokeh')['RGB'][group].allowed_keywords
]
valid_rgb_options.remove(
    'alpha')  # remove option set by sliders on individual channels

# TODO move to color module
import colorcet as cc

# repeat colormap to handle unint16 values
# needed to handle non continuous labels because colormap is stretched (and not cycled)
label_cmap = cc.b_glasbey_hv * 256


# bokeh hook workaround --> remove if holoviews finally handle this
def zoom_bounds_hook(bounds):
    '''restrict zooming out to given bounds'''
    def _hook(plot, element):
        plot.state.x_range.bounds = (bounds[0], bounds[2])
        plot.state.y_range.bounds = (bounds[1], bounds[3])
        plot.state.select(WheelZoomTool).maintain_focus = False

    return _hook


def get_img_dims_coords(img, spacing=1):

    img_dims = ['x', 'y', 'z'][:img.ndim]
    spacing = np.broadcast_to(np.array(spacing), img.ndim)
    img_coords = [
        np.arange(d) * s for d, s in zip(img.shape[::-1], spacing[::-1])
    ]

    return img_dims, img_coords


def image_to_hvds(img, label, spacing=1):
    '''Converts a 2D/3D image to a holoview dataset to facilitate
    plotting with the correct axis bounds/scaling'''

    img_dims, img_coords = get_img_dims_coords(img, spacing)

    return hv.Dataset((*(img_coords), img),
                      kdims=img_dims,
                      vdims=['intensity'],
                      label=label)


class HvDataset(param.Parameterized):
    '''Converts a numpy image to holoviews Dataset dynamic map'''

    img = param.Array(np.zeros((2, 2), dtype=np.uint8),
                      doc='numpy iamge array',
                      precedence=-1)
    label = param.String('channel',
                         doc='label for the generated hv.Dataset',
                         precedence=-1)
    spacing = param.Parameter((1, ), doc='pixel/voxel size', precedence=-1)

    _update_counter = param.Integer(0, precedence=-1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._broadcast_spacing()

    @param.depends()
    def _broadcast_spacing(self):
        self.spacing = tuple(
            np.broadcast_to(np.array(self.spacing), self.img.ndim).tolist())

    @param.depends('img', watch=True)
    def _update_img(self):
        self._broadcast_spacing()
        self._update_counter += 1

    # NOTE dynamic map with dependency directly on array is less responsive (hash computation overhead?)
    @param.depends('_update_counter', 'label')
    def _build_dataset(self):
        return image_to_hvds(self.img, self.label, self.spacing)

    @param.depends('spacing')
    def dmap(self):
        return hv.DynamicMap(self._build_dataset, cache_size=1)


def make_composite(imgs, cmaps, mode='max'):
    '''embeds colormap and blend grescale input images into a rgb image'''

    _modes = {'max': np.max, 'mean': np.mean}

    blending_fun = _modes.get(mode, None)

    if blending_fun is None:
        raise NotImplementedError(
            'blending mode note implemented: {}'.format(mode))

    imgs = [(plt.get_cmap(name)(img)[..., :-1] * 255).astype(np.uint8)
            for img, name in zip(imgs, cmaps)]

    blended_img = blending_fun(np.asarray(imgs), axis=0)
    return np.rint(blended_img).astype(np.uint8)


def blend_overlay(elems):
    '''Transforms a hv.Overlay of hv.Image into a hv.RGB'''

    if not isinstance(elems, hv.Overlay):
        # probably a single channel, do nothing
        return elems

    imgs = [e.dimension_values(2, flat=False) for e in elems]

    if imgs[0].dtype != np.uint8:
        raise ValueError(
            '8 bit images are expected to stack overlays, got {}'.format(
                imgs[0].dtype))

    # embed colormap,opacity and blend
    # Note somehow hv.RGB inverts the y axis but not hv.Image???
    cmaps = [e.opts.get().options['cmap'] for e in elems]
    alphas = [e.opts.get().options['alpha'] for e in elems]
    imgs = [(a * img).astype(int) if a < 1.0 else img
            for a, img in zip(alphas, imgs)]
    rgb = make_composite(imgs, cmaps, mode='max')[::-1]

    xr = elems.range(0)
    yr = elems.range(1)
    bounds = (xr[1], yr[0], xr[0], yr[1])
    height, width = rgb.shape[:-1]

    options = list(elems)[0].opts.get().options
    options = {
        key: val
        for key, val in options.items() if key in valid_rgb_options
    }

    return hv.RGB(rgb, bounds=bounds, group='composite').opts(**options)


def split_element(element, axis, values=None):
    '''Applies element.select to all values along axis and returns the result as a list.
    
    Dimension values can also be specified explicitly to select a subset or control the order.'''

    new_dims_name = [d.name for d in element.kdims if d.name != axis]
    if values is None:
        values = element.dimension_values(axis, expanded=False)

    return tuple(
        element.select(**{
            axis: val
        }).reindex(new_dims_name).relabel(val) for val in values)
