import numpy as np
from collections import OrderedDict

import holoviews as hv
hv.extension('bokeh', logo=False)

from holoviews.operation.datashader import rasterize, regrid
import datashader

import param
import panel as pn


def image_to_hvds(imgs, spacing=1, keys=None):
    '''Converts a given image to a holoview dataset to facilitate
    plotting with the correct axis bounds/scaling'''

    if keys is None:
        keys = list(imgs.keys())
    imgs = [imgs[key] for key in keys]
    imgs = [img for img in imgs]

    img = np.stack(imgs, axis=0)

    # get image dims and coords from first image in list
    img_dims = ['x', 'y', 'z'][:imgs[0].ndim]
    spacing = np.broadcast_to(spacing, imgs[0].ndim)
    img_coords = [
        np.arange(d) * s for d, s in zip(imgs[0].shape[::-1], spacing[::-1])
    ]

    # ds extra dataset coords
    #     ds_coords = [keys]
    ds_coords = [hv.Dimension('ch', label='ch', values=keys)]

    return hv.Dataset((*(img_coords + ds_coords), img),
                      kdims=img_dims + ['ch'],
                      vdims=['intensity'])


def image_to_rgb_hvds(imgs, spacing=1, keys=None):
    '''Converts a given image to a holoview dataset to facilitate
    plotting with the correct axis bounds/scaling'''

    if keys is None:
        keys = list(imgs.keys())
    imgs = tuple(imgs[key] for key in keys)
    #     img = tuple(img for img in imgs)

    #     img = np.stack(imgs, axis=-1).view(dtype=[('r', np.uint8), ('g', np.uint8), ('b', np.uint8)]).squeeze()

    # get image dims and coords from first image in list
    img_dims = ['x', 'y', 'z'][:imgs[0].ndim]
    spacing = np.broadcast_to(spacing, imgs[0].ndim)
    img_coords = [
        np.arange(d) * s for d, s in zip(imgs[0].shape[::-1], spacing[::-1])
    ]

    return hv.Dataset((*(img_coords), *imgs),
                      kdims=img_dims,
                      vdims=list('rgb'),
                      label='rgb')


class UpdatableOperation(param.Parameterized):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, element):
        return hv.util.Dynamic(element,
                               operation=self._dynamic_call,
                               shared_data=False)

    @param.depends()
    def _dynamic_call(self, element):
        return element

    @property
    def widget(self):
        # only return the widget, without param title (label is already displayed)
        return pn.Param(self.param)[1]


class Alpha(UpdatableOperation):
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

    def update_coords(self, coords):
        if not np.array_equal(self.param.slice_id.objects, coords):
            self.param.slice_id.objects = coords
            self.slice_id = coords[len(coords) // 2]

    @param.depends('slice_id')
    def _dynamic_call(self, element):
        self.update_coords(element.dimension_values(self.axis, expanded=False))
        new_dims_name = [d.name for d in element.kdims if d.name != self.axis]

        return element.select(**{
            self.axis: self.slice_id
        }).reindex(new_dims_name)

    @property
    def widget(self):
        return pn.Param(self.param,
                        widgets={
                            'slice_id': {
                                'type': pn.widgets.DiscreteSlider,
                                'formatter': '%.2g'
                            }
                        })[1]


def split_element(element, axis, values=None):
    '''Applies element.select to all values along axis and returns the result as a list.
    
    Dimension values can also be specified explicitly to select a subset or control the order.'''

    new_dims_name = [d.name for d in element.kdims if d.name != axis]
    if values is None:
        values = element.dimension_values(axis, expanded=False)

    return OrderedDict((val, element.select(**{
        axis: val
    }).reindex(new_dims_name).relabel(val)) for val in values)
    # ~return OrderedDict((val,element.select(**{axis:val}).reindex(new_dims_name).relabel(val)) for val in values)


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


class flip_axis(hv.Operation):
    axis = param.String()

    def _process(self, element, key=None):

        # this works for the channel axis (if exist) but not image axis????
        #         vals = element.dimension_values(self.p.axis, expanded=False)
        #         return element.redim.values(**{self.p.axis:vals[::-1]})

        if isinstance(element, (hv.Image, hv.RGB)):
            element.data[self.p.axis][:] = element.data[self.p.axis][::-1]

        elif isinstance(element, (hv.HoloMap, hv.Overlay)):
            for e in element:
                flip_axis(e, axis=self.p.axis)

        else:
            raise ValueError(
                'flip_axis not implement for element of type: {}'.format(
                    type(element)))

        return element


# TODO test with larger image+larger window
# rasterize method that use a different aggregate method for annotation keys
# TODO pass label keys in annot (handle capitatlization)
class rasterize_overlay(rasterize):
    def _process(self, element, key=None):

        # TODO test overlay type

        for (group, label), sub_element in element.data.items():

            #             display(self._aggregator_param_value, self.aggregator)

            if label in ['Annot', 'Pred']:
                #                 print('aaaaa', group, label)
                self.p.aggregator = datashader.first()
                element.data[(group, label)] = super()._process(sub_element)
            else:
                self.p.aggregator = datashader.mean()
                element.data[(group, label)] = super()._process(sub_element)

        return element


# rasterize with custom aggreagtion methods for labels e.g 'first'
# ~class regrid_custom_agg(regrid):
# ~'''Maps element label to custom aggregation method if specified'''

# ~agg_mapping = param.Dict()

# ~def _get_aggregator(self, element, add_field=True):

# ~agg_mapping = self.p.agg_mapping.get(element.label)
# ~if agg_mapping:
# ~agg = self._agg_methods[agg_mapping]()
# ~else:
# ~agg = super()._get_aggregator(element, add_field)

# ~return agg

# ~class rasterize_custom_agg(rasterize):
# ~'''Replaces regrid by regrid_custom_agg for hv.Image'''

# ~agg_mapping = param.Dict()

# TODO find a way to do it without breaking normal rasterize
# replacing _transforms at instance level in init does not seem to work
# ~rasterize_custom_agg._transforms[0] = (hv.Image, regrid_custom_agg)
