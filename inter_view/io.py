import numpy as np
import pandas as pd

import holoviews as hv
hv.extension('bokeh', logo=False)

import param
import panel as pn


# TODO dynamic widget box showing only available sublevels
class DataHandler(param.Parameterized):
    dc = param.DataFrame()

    def __init__(self, dc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dc = dc
        self._add_widgets()

    def _add_widgets(self):
        if isinstance(self.dc.index, pd.core.index.MultiIndex):
            self.wg = [
                pn.widgets.Select(value=self.dc.reset_index().loc[0, name],
                                  options=vals.tolist(),
                                  name=name) for name, vals in
                zip(self.dc.index.names, self.dc.index.levels)
            ]
        else:
            self.wg = [
                pn.widgets.Select(
                    value=self.dc.reset_index().loc[0, self.dc.index.name],
                    options=self.dc.index.tolist(),
                    name=self.dc.index.name)
            ]

    def view(self):
        @pn.depends(*[w.param.value for w in self.wg])
        def inner_view(*args):
            try:
                if len(args) == 1:  # single level index, use key directly
                    args = args[0]
                return hv.Table(
                    self.dc.lsc.__getitem__(args).reset_index()).opts(
                        width=900)
            except KeyError as e:
                return 'invalid index'

        return inner_view

    @property
    def layout(self):
        return pn.Column(pn.WidgetBox(*self.wg), self.view())


# TODO handle additional axis
class ImageHandler(DataHandler):
    ds_dims = param.List(
        doc='extra dimensions to be included in hv.dataset. e.g. image channels'
    )
    ds = param.parameterized.Parameter()
    ds_id = param.Number(
        0, doc="""Random id assigned every time a new hv.Dataset is loaded""")
    spacing = param.Array(np.array([1]))

    def __init__(self, dc, ds_dims=None, spacing=1, *args, **kwargs):
        if ds_dims:
            dc = dc.reset_index(ds_dims)

        super().__init__(dc, *args, **kwargs)

        self.spacing = np.asarray(spacing)
        if ds_dims:
            self.ds_dims = ds_dims
        self.load()(*[w.value for w in self.wg])

    @param.depends()
    def _image_to_hvds(self, imgs):
        '''Converts a given image to a holoview dataset to facilitate
        plotting with the correct axis bounds/scaling'''

        if len(self.ds_dims) > 1:
            # TODO reshape array after stacking (also don't rely on data collection having levels, i.e. multiindex)
            raise NotImplementedError(
                'dataset with more than 1 dim not implemented, dims {}'.format(
                    self.ds_dims))
        img = np.stack(imgs, axis=0).squeeze()

        # get image dims and coords from first image in list
        img_dims = ['x', 'y', 'z'][:imgs[0].ndim]
        spacing = np.broadcast_to(self.spacing, imgs[0].ndim)
        img_coords = [
            np.arange(d) * s
            for d, s in zip(imgs[0].shape[::-1], spacing[::-1])
        ]

        # ds extra dataset coords
        ds_coords = [self.dc[d].unique().tolist() for d in self.ds_dims]

        return hv.Dataset((*(img_coords + ds_coords), img),
                          kdims=img_dims + self.ds_dims,
                          vdims=['intensity'])

    def load(self):
        @pn.depends(*[w.param.value for w in self.wg], watch=True)
        def inner_load(*args):
            try:
                if len(args) == 1:  # single level index, use key directly
                    args = args[0]
                dc = self.dc.lsc.__getitem__(args).reset_index()  #.iloc[0,:]
                imgs = dc.lsc.read()
                self.ds = self._image_to_hvds(imgs)
                self.ds_id = np.random.rand()

            except KeyError as e:
                self.ds = hv.Dataset([])

        return inner_load
