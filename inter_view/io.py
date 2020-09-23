import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import gc
import param
import panel as pn
import holoviews as hv
hv.extension('bokeh', logo=False)

from improc.io import DCAccessor
DCAccessor.register()


class CollectionHandler(param.Parameterized):
    '''Builds an interactive menu matching the index/multi-index of a dataframe. 
    The indexed dataframe is available as "subdf" attribute.'''

    df = param.DataFrame()
    subdf = param.DataFrame()
    file_widgets = param.List([])

    updating_widget = param.Boolean(False)
    update_count = param.Integer(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_widgets()

    def _get_available_index_level_values(self, level_id):
        if level_id == 0:
            index = self.df.index
            if index.name is None:
                index.name = 'index'
        else:
            lower_levels_index = tuple(w.value
                                       for w in self.file_widgets[:level_id])
            index = self.df.dc[lower_levels_index].index

        return index.unique(level=level_id)

    def _get_level_widget(self, level_id):

        options = self._get_available_index_level_values(level_id)

        # TODO mouseup callback policy
        # value_throttle is not updating, unlike mentioned in the doc, callback_policy is not a valid option
        # TODO init value_throttled, dublicate options if only one
        #         if np.issubdtype(level, np.number):
        #             return pn.widgets.DiscreteSlider(name=level.name, value=level[0], options=level.tolist())
        #         else:
        return pn.widgets.Select(name=options.name,
                                 value=options[0],
                                 options=options.tolist())

    def _add_widget_watcher(self):
        @pn.depends(*[w.param.value for w in self.file_widgets], watch=True)
        def _widget_watcher(*args):

            if not self.updating_widget:
                # ignore changes until update is finished
                self.updating_widget = True

                for level_id, wg in enumerate(self.file_widgets):
                    wg.options = self._get_available_index_level_values(
                        level_id).tolist()

                self.updating_widget = False
                self.update_count += 1

                idx = tuple(w.value for w in self.file_widgets)
                if len(idx) == 1:
                    idx = idx[0]

                self.subdf = self.df.dc[idx]

        _widget_watcher()

    def _build_widgets(self):
        index = self.df.index

        if isinstance(index, pd.MultiIndex):
            for lvl_id in range(len(index[0])):
                self.file_widgets.append(self._get_level_widget(lvl_id))

        else:
            self.file_widgets.append(self._get_level_widget(0))

        self._add_widget_watcher()

    @param.depends('subdf')
    def view_df(self):
        return hv.Table(self.subdf.reset_index()).opts(width=900)

    def widgets(self):
        return pn.WidgetBox(*self.file_widgets)

    def panel(self):
        return pn.Column(self.widgets, self.view_df)


class MultiCollectionHandler(CollectionHandler):
    '''Variant of DataHandler that allow multiple selection for certain index levels'''

    multi_select_levels = param.List([])
    _multi_select_levels_ids = param.List()

    def _get_level_widget(self, level_id):
        if not self._multi_select_levels_ids:
            names = self.df.index.names
            self._multi_select_levels_ids = [
                names.index(l_name) for l_name in self.multi_select_levels
            ]

        options = self._get_available_index_level_values(level_id)

        if level_id in self._multi_select_levels_ids:
            wg = pn.widgets.MultiSelect(
                name=options.name,
                value=options.tolist(),
                options=options.tolist(),
                # ~height_policy='fit',
                size=min(len(options.tolist()), 10))

        else:
            wg = pn.widgets.Select(name=options.name,
                                   value=options[0],
                                   options=options.tolist())
        return wg

    def multi_select_index(self):
        '''returns multi select index level of current sub collection'''
        return tuple(
            tuple(self.subdf.index.get_level_values(l))
            for l in self.multi_select_levels)


class DataLoader(MultiCollectionHandler):
    '''Extension of MultiCollectionHandler that automatically load files of selected part of the data collection'''

    loaded_objects = param.Dict()
    loading_fun = param.Callable(
        doc=
        'callable loading function accepting a path as argument. If None the pandas DC accessor is used'
    )
    loading_off = param.Boolean(False)

    @param.depends('subdf', watch=True)
    def _load_files(self):

        if not self.loading_off:
            if self.loading_fun is not None:
                # read files in parallel
                with ThreadPoolExecutor() as threads:
                    files = threads.map(self.loading_fun, self.subdf.dc.path)
            else:
                files = self.subdf.dc.read()

            # set keys using levels that can have multiple selection
            if len(self.multi_select_levels) == 0:
                keys = self.subdf.index
            elif len(self.multi_select_levels) == 1:
                keys = self.subdf.index.get_level_values(
                    self.multi_select_levels[0])
            else:
                keys = zip(*[
                    self.subdf.index.get_level_values(l)
                    for l in self.multi_select_levels
                ])

            self.loaded_objects = {k: f for k, f in zip(keys, files)}

            # force garbage collection
            gc.collect()
