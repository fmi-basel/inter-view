import numpy as np

import holoviews as hv
hv.extension('bokeh', logo=False)

from holoviews.operation.datashader import rasterize
from bokeh.models import WheelZoomTool

import param
import panel as pn
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
    spacing = np.broadcast_to(spacing, img.ndim)
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


def image_to_rgb_hvds(imgs, spacing=1, keys=None, label=None):
    '''Converts a 2D/3D RGB channels to a holoview dataset to facilitate
    plotting with the correct axis bounds/scaling
    '''

    if label is None:
        label = 'rgb'

    # if now keys given, assumes there are 3|4 channels in the right order
    if keys is None:
        keys = list(imgs.keys())

    imgs = tuple(imgs[key] for key in keys)

    img_dims, img_coords = get_img_dims_coords(imgs[0], spacing)

    return hv.Dataset((*(img_coords), *imgs),
                      kdims=img_dims,
                      vdims=keys,
                      label=label)


class UpdatableOperation(param.Parameterized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, element):
        return hv.util.Dynamic(element, operation=self._dynamic_call).relabel(
            element.label)

    @param.depends()
    def _dynamic_call(self, element):
        return element

    @property
    def widget(self):
        # only return the widget, without param title (label is already displayed)
        return pn.Param(self.param)[1]


class Alpha(UpdatableOperation):

    # see OverlayViewer for a javascript more responsive callback
    alpha = param.Number(default=1.0, bounds=(0., 1.0), step=0.01)

    def __init__(self, label='alpha', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param.alpha.label = label

    @param.depends('alpha')
    def _dynamic_call(self, element):
        return element.opts(alpha=self.alpha)


class Slice(UpdatableOperation):
    slice_id = param.ObjectSelector(default=0, objects=[0])
    axis = param.String(default='z')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.param.slice_id.label = self.axis
        self.initialized = False
        self._widget = pn.Param(self.param,
                                widgets={
                                    'slice_id': {
                                        'type': pn.widgets.DiscreteSlider,
                                        'formatter': '%.2g'
                                    }
                                })[1]

    def reset(self):
        self.initialized = False

    def update_coords(self, element):
        coords = element.dimension_values(self.axis, expanded=False)
        self.param.slice_id.objects = coords
        self.slice_id = coords[len(coords) // 2]

    def _find_nearest_value(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def moveto(self, value):
        '''Move slider to it's closest (discrete) position'''

        discrete_values = self.param.slice_id.objects
        self.slice_id = self._find_nearest_value(discrete_values, value)

    def __call__(self, element):
        if not self.initialized:

            if isinstance(element, hv.DynamicMap):
                # "render" a dummy copy of the DynamicMap  to evaluate and get the slider range
                try:
                    pn.Row(super().__call__(element.clone())).embed()
                except Exception as e:
                    pass

                self.initialized = True
            else:
                self.update_coords(element)
                self.initialized = True

        return super().__call__(element)

    @param.depends('slice_id')
    def _dynamic_call(self, element):
        if not self.initialized:
            self.update_coords(element)

        new_dims_name = [d.name for d in element.kdims if d.name != self.axis]

        # since we are reindexing anyway, flip axis if y-z plane to correct orientation in orthoview
        # plot can also be flipped afterwards with invert_axes=True but creates BUG when calling "datashader/rasterize"
        if tuple(new_dims_name) == ('y', 'z'):
            new_dims_name = ['zb', 'y']
            # rename dim to avoid conflict between projections with z axis vertical/horizontal
            element = element.redim(z='zb')

        return element.select(**{
            self.axis: self.slice_id
        }).reindex(new_dims_name)

    @property
    def widget(self):
        return self._widget


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


# slow, faster to build rgb dataset directly
class format_as_rgb(hv.Operation):
    dim = param.parameterized.Parameter('None')
    normalize_separate = param.Boolean(
        True, doc="""Whether or not to normalize the channels separately""")

    ch_order = param.List(default=[], doc="""list of Channels to use as rgb""")

    def _min_max_scaling(self, img, ch_axis=0):
        if self.p.normalize_separate:
            axis = tuple(ax for ax in range(img.ndim) if ax != ch_axis)
            keepdims = True
        else:
            axis = None
            keepdims = False

        img -= img.min(axis=axis, keepdims=keepdims)
        img = img / (img.max(axis=axis, keepdims=keepdims) + 1e-5)
        return (img * 255).astype(np.uint8)

    @staticmethod
    def _reverse_axis_order(arr):
        order = list(range(arr.ndim))
        return np.moveaxis(arr, order, order[::-1])

    def _process(self, element, key=None):

        dims = element.dimensions()[:-1]
        dims_name = [d.name for d in dims]
        dims_value = [
            element.dimension_values(name, expanded=False)
            for name in dims_name
        ]
        dims_len = tuple(len(arr) for arr in dims_value)

        dim_idx = dims_name.index(self.p.dim)

        # if ch order not provided, use first 3
        if not self.p.ch_order:
            self.p.ch_order = dims_value[dim_idx].tolist()[:3]

        # find corresponding ch indices
        rgb_idxs = [
            dims_value[dim_idx].tolist().index(name)
            for name in self.p.ch_order
        ]

        # NOTE because of bug in dimension_values(flat=False), reshaping needs to be done manually
        # https://github.com/pyviz/holoviews/issues/4054
        rgb_data = element.dimension_values(len(dims)).reshape(dims_len)

        rgb_data = np.take(
            rgb_data, rgb_idxs,
            axis=dim_idx)  # in case there is more than 3, keep first 3
        rgb_data = self._min_max_scaling(rgb_data, ch_axis=dim_idx)

        rgb_data = [np.take(rgb_data, ch, axis=dim_idx) for ch in range(3)]
        rgb_data = [self._reverse_axis_order(arr) for arr in rgb_data]

        axis_data = [
            val for idx, val in enumerate(dims_value) if idx != dim_idx
        ]

        kdims = [name for idx, name in enumerate(dims_name) if idx != dim_idx]
        vdims = self.p.ch_order

        if len(kdims) == 2:
            return hv.RGB(tuple(axis_data + rgb_data),
                          kdims=kdims,
                          vdims=vdims)
        else:
            # too many dimension for an image, return a dataset with rgb channels as vdims
            return hv.Dataset(tuple(axis_data + rgb_data),
                              kdims=kdims,
                              vdims=vdims)

        return element


class label_overlay_items(hv.Operation):
    '''labelled overlayed images so that clickable legend can be displayed'''
    def _process(self, element, key=None):

        if isinstance(element, (hv.Overlay, hv.NdOverlay)):

            name = element.dimensions()[0].name
            labels = element.dimension_values(0)

            items = {
                label: item.relabel(label)
                for label, item in zip(labels, element)
            }
            return type(element)(items, kdims=[name])

        else:
            return element


def rasterize_custom(elem, label_channels, dynamic=True):
    '''applies rasterization with "first" aggregation if elem.label 
    in label_channels, default aggregation otherwise.'''

    aggregator = 'default'
    if elem.label in label_channels:
        aggregator = 'first'

    return rasterize(elem, aggregator=aggregator, dynamic=dynamic)
