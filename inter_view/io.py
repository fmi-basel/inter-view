import numpy as np
import pandas as pd

import holoviews as hv
hv.extension('bokeh', logo=False)

import param
import panel as pn


class DataHandler(param.Parameterized):
    dc = param.DataFrame()
    subdc = param.DataFrame()
    widgets = param.List([])

    updating_widget = param.Boolean(False)
    update_count = param.Integer(0)

    def __init__(self, dc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dc = dc
        self._add_widgets()

    def _get_available_index_level_values(self, level_id):
        if level_id == 0:
            index = self.dc.index
            if index.name is None:
                index.name = 'index'
        else:
            lower_levels_index = [w.value for w in self.widgets[:level_id]]
            index = self.dc.lsc[lower_levels_index].index

        return index.get_level_values(level_id).unique()

    def _get_level_widget(self, level_id):

        level = self._get_available_index_level_values(level_id)

        # TODO mouseup callback policy
        # seems currently broken in panel, undocumented value_throttle option that doens't do anything
        #         if np.issubdtype(level, np.number):
        #             return pn.widgets.DiscreteSlider(name=level.name, value=level[0], options=level.tolist())
        #         else:
        return pn.widgets.Select(name=level.name,
                                 value=level[0],
                                 options=level.tolist())

    def _add_widget_watcher(self):
        @pn.depends(*[w.param.value for w in self.widgets], watch=True)
        def _widget_watcher(*args):

            if not self.updating_widget:
                # ignore changes until update is finished
                self.updating_widget = True

                for level_id, wg in enumerate(self.widgets):
                    wg.options = self._get_available_index_level_values(
                        level_id).tolist()

                self.updating_widget = False
                self.update_count += 1

                idx = tuple(w.value for w in self.widgets)
                if len(idx) == 1:
                    idx = idx[0]

                self.subdc = self.dc.lsc[idx].reset_index()

        _widget_watcher()

    def _add_widgets(self):
        index = self.dc.index

        if isinstance(index, pd.MultiIndex):
            for lvl_id in range(len(index[0])):
                self.widgets.append(self._get_level_widget(lvl_id))

        else:
            self.widgets.append(self._get_level_widget(0))

        self._add_widget_watcher()

    @param.depends('subdc')
    def view_dc(self):
        return hv.Table(self.subdc).opts(width=900)

    @property
    def widgetbox(self):
        return pn.WidgetBox(*self.widgets)

    @property
    def panel(self):
        return pn.Column(self.widgetbox, self.view_dc)


class ImageHandler(DataHandler):
    ds_dims = param.List(
        doc=
        """extra dimensions to be included in hv.dataset. e.g. image channels"""
    )
    ds = param.Parameter()

    #     ds_id = param.Number(0, doc="""Random id assigned every time a new hv.Dataset is loaded""")

    load_count = param.Integer(
        0,
        doc="""counter incrementing every time a new hv.Dataset is loaded""")
    spacing_col = param.Parameter(
        None, doc="""name of column containing image spacing""")

    #     spacing = param.Array(np.array([1]))

    def __init__(self, dc, ds_dims=[], *args, **kwargs):
        if ds_dims:
            try:
                dc = dc.reset_index(ds_dims)
            except Exception as e:
                pass

        super().__init__(dc, *args, ds_dims=ds_dims, **kwargs)


#         self.load()(*[w.value for w in self.wg])

    @param.depends()
    def _image_to_hvds(self, imgs):
        '''Converts images to a holoview dataset to facilitate
        plotting with the correct axis bounds/scaling'''

        # TODO test more dimensions (time?)

        if self.spacing_col is None:
            spacing = 1
        else:
            spacing = self.subdc.iloc[0][self.spacing_col]

        img = np.stack(imgs, axis=0).squeeze()

        # get image dims and coords from first image in list
        img_dims = ['x', 'y', 'z'][:imgs[0].ndim]
        spacing = np.broadcast_to(spacing, imgs[0].ndim)
        img_coords = [
            np.arange(d) * s
            for d, s in zip(imgs[0].shape[::-1], spacing[::-1])
        ]

        # ds extra dataset coords
        ds_coords = [self.subdc[d].values.tolist() for d in self.ds_dims]

        return hv.Dataset((*(img_coords + ds_coords), img),
                          kdims=img_dims + self.ds_dims,
                          vdims=['intensity'])

    @param.depends('subdc', watch=True)
    def update_ds(self):
        imgs = self.subdc.lsc.read()
        self.ds = self._image_to_hvds(imgs)
        self.load_count += 1
