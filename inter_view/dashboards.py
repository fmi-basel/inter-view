import numpy as np
import os
import pandas as pd
import param
import panel as pn
import holoviews as hv
from collections.abc import Iterable

from holoviews import opts, dim
from holoviews.streams import Selection1D
from bokeh.models import HoverTool

from inter_view.utils import HvDataset  #, label_cmap, split_element, zoom_bounds_hook# rasterize_custom
from inter_view.view_images import SliceViewer, OverlayViewer, OrthoViewer, CompositeViewer, SegmentationViewer
from inter_view.edit_images import RoiEditor, EditableHvDataset, FreehandEditor
from inter_view.io import DataLoader, MultiCollectionHandler


class BaseImageDashBoard(DataLoader):
    '''Base class to build image related dashboards.
    
    Maintains 2 counter attributes indicating whether partial dynamic 
    updates are sufficient. Complete updates are considered necessary if
    the image shape has changed (i.e. requires rebuilding a plot with new 
    bounds, aspect ratio, etc) or if the multiselection as changed (e.g. 
    a different set of channels is selected)'''

    _dynamic_update_counter = param.Integer(0)
    _complete_update_counter = param.Integer(0)

    _has_shape_changed = param.Boolean(True)
    _has_multiselect_changed = param.Boolean(True)

    _old_img_shape = param.Parameter((-1, ))
    _old_multi_select_index = param.Parameter((-1, ))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._monitor_updates()

    @staticmethod
    def index_to_str(index):
        '''Converts a tuple index to a string wit _ separators'''
        if not isinstance(index, (tuple, list)):
            return str(index)
        else:
            index = map(str, index)
            return '_'.join(index)

    @param.depends('loaded_objects', watch=True)
    def _monitor_updates(self):
        # reset
        self._has_shape_changed = False
        self._has_multiselect_changed = False

        # check shape of any loaded images (assume the rest is the same)
        img_shape = next(iter(self.loaded_objects.values())).shape

        # check multi selection has changed (e.g. different set of channels)
        multi_select_index = self.multi_select_index()

        if img_shape != self._old_img_shape:
            self._old_img_shape = img_shape
            self._has_shape_changed = True

        if multi_select_index != self._old_multi_select_index:
            self._old_multi_select_index = multi_select_index
            self._has_multiselect_changed = True

        if self._has_shape_changed or self._has_multiselect_changed:
            self._complete_update_counter += 1
        else:
            self._dynamic_update_counter += 1


class CompositeDashBoard(BaseImageDashBoard):
    '''Dashboard to views 2D, multi-channel images as color composite.'''

    channel_config = param.Dict({}, doc='dictionnary configuring each channel')
    composite_viewer = param.Parameter(CompositeViewer())
    hv_datasets = param.List()
    slicer = param.Parameter(SliceViewer())

    _widget_update_counter = param.Integer(0)

    @param.depends('_dynamic_update_counter', watch=True)
    def _dynamic_img_update(self):
        for hv_ds, img in zip(self.hv_datasets, self.loaded_objects.values()):
            hv_ds.img = img

    @param.depends('_complete_update_counter')
    def dmap(self):

        if not self.composite_viewer.channel_viewers or self._has_multiselect_changed:
            selected_channel_config = {
                key: self.channel_config[key]
                for key in self.loaded_objects.keys()
            }
            self.composite_viewer = CompositeViewer.from_channel_config(
                selected_channel_config)
            self._widget_update_counter += 1

        self.hv_datasets = [
            HvDataset(img=img, label=self.index_to_str(key))
            for key, img in self.loaded_objects.items()
        ]
        dmaps = [hv_ds.dmap() for hv_ds in self.hv_datasets]

        # apply slicer if 3d image
        if next(iter(self.loaded_objects.values())).ndim > 2:
            dmaps = [self.slicer(dmap) for dmap in dmaps]

        dmap = self.composite_viewer(dmaps)

        return dmap

    @param.depends('_widget_update_counter')
    def widgets(self):

        wg = [super().widgets(), self.composite_viewer.widgets]
        wg = list(filter(None, wg))
        return pn.Column(*wg)

    def panel(self):
        if list(self.loaded_objects.values())[0].ndim > 2:
            plot = self.slicer.panel(self.dmap)
        else:
            plot = self.dmap

        return pn.Row(plot, self.widgets)


class ExportCallback(param.Parameterized):
    '''Mixin class adding channel export callbacks'''

    out_folder = param.String('', doc='output folder')
    export_funs = param.List(
        doc=
        'list of callables with signature (path, imgs, cmaps, intensity_bounds, labels)'
    )
    export_fun = param.ObjectSelector(doc='active export function')
    export_viewers = param.Dict({})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.export_funs:
            raise ValueError('At least one export function is required')
        self.param.export_fun.objects = self.export_funs
        self.export_fun = self.export_funs[0]

    @staticmethod
    def _validate_intensity_bounds(intensity_bounds, img):
        if intensity_bounds is None:
            return img.min(), img.max()
        else:
            return intensity_bounds

    def _get_export_arguments(self):
        imgs = list(self.loaded_objects.values())
        labels = list(self.export_viewers.keys())
        intensity_bounds = [
            self._validate_intensity_bounds(v.intensity_bounds, img)
            for v, img in zip(self.export_viewers.values(), imgs)
        ]
        cmaps = [v.cmap for v in self.export_viewers.values()]
        in_path = self.subdf.iloc[0:1].dc.path.tolist()[0]
        out_path = os.path.join(self.out_folder, os.path.basename(in_path))
        return out_path, imgs, cmaps, intensity_bounds, labels

    def _export_callback(self, event):
        args = self._get_export_arguments()
        self.export_fun(*args)

    def _export_widgets(self):
        export_button = pn.widgets.Button(name='export')
        export_button.on_click(self._export_callback)
        return pn.WidgetBox(
            pn.Param(self.param.export_fun,
                     widgets={'export_fun': {
                         'name': ''
                     }}), export_button)


class RoiExportCallback(ExportCallback, RoiEditor):
    def _get_export_arguments(self):
        out_path, imgs, cmaps, intensity_bounds, labels = super(
        )._get_export_arguments()

        if imgs[0].ndim > 2:
            axis = {'z': 0, 'y': 1, 'x': 2}[self.slicer.axis]
            imgs = [
                np.take(img, self.slicer.slice_id, axis=axis) for img in imgs
            ]
            pre, ext = os.path.splitext(out_path)
            out_path = pre + '_z{}'.format(self.slicer.slice_id) + ext

        # crop images if roi available and add roi bounds to filename
        loc = self.img_slice()
        if loc is not None:
            # only exports first roi
            loc = loc[0]
            imgs = [img[loc] for img in imgs]
            pre, ext = os.path.splitext(out_path)
            roi_bounds = (loc[1].start, loc[1].stop, loc[0].start, loc[0].stop)
            out_path = pre + '_roi_{}_{}_{}_{}'.format(*roi_bounds) + ext

        return out_path, imgs, cmaps, intensity_bounds, labels


class CompositeExportDashBoard(CompositeDashBoard, RoiExportCallback):
    '''Extension of CompositeDashboard with figure export capabilities'''
    @param.depends('_complete_update_counter')
    def dmap(self):
        dmap = super().dmap()
        self.export_viewers = self.composite_viewer.channel_viewers

        return dmap * self.roi_plot

    @param.depends('_complete_update_counter')
    def widgets(self):
        wg = [super().widgets(), self._export_widgets()]
        return pn.Column(*wg)


class SegmentationDashBoard(BaseImageDashBoard):
    '''Dashboard to views 2D, multi-channel images as color composite.'''

    channel_config = param.Dict({}, doc='dictionnary configuring each channel')
    composite_channels = param.List(
        doc='ids of channels to be displayed as color composite')
    overlay_channels = param.List(
        doc='ids of channels to be displayed as overlay on top of the composite'
    )
    segmentation_viewer = param.Parameter(SegmentationViewer())
    hv_datasets = param.List()
    slicer = param.Parameter(SliceViewer())

    _widget_update_counter = param.Integer(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @param.depends('_dynamic_update_counter', watch=True)
    def _dynamic_img_update(self):
        for hv_ds, img in zip(self.hv_datasets, self.loaded_objects.values()):
            hv_ds.img = img

    def _get_channel_dmaps(self):
        '''Returns 1 dynamic map per channel (i.e. not overlayed or blended as composite)'''

        if not self.segmentation_viewer.channel_viewers or self._has_multiselect_changed:
            selected_channel_config = {
                key: self.channel_config[key]
                for key in self.loaded_objects.keys()
            }
            self.segmentation_viewer = SegmentationViewer.from_channel_config(
                selected_channel_config,
                composite_channels=self.composite_channels,
                overlay_channels=self.overlay_channels)
            self._widget_update_counter += 1

        self.hv_datasets = [
            HvDataset(img=img, label=self.index_to_str(key))
            for key, img in self.loaded_objects.items()
        ]
        dmaps = [hv_ds.dmap() for hv_ds in self.hv_datasets]

        # apply slicer if 3d image
        if next(iter(self.loaded_objects.values())).ndim > 2:
            dmaps = [self.slicer(dmap) for dmap in dmaps]

        return dmaps

    @param.depends('_complete_update_counter')
    def dmap(self):
        channel_dmaps = self._get_channel_dmaps()
        dmap = self.segmentation_viewer(dmaps)

        return dmap

    @param.depends('_widget_update_counter')
    def widgets(self):

        wg = [super().widgets(), self.segmentation_viewer.widgets]
        wg = list(filter(None, wg))
        return pn.Column(*wg)

    def panel(self):
        if list(self.loaded_objects.values())[0].ndim > 2:
            plot = self.slicer.panel(self.dmap)
        else:
            plot = self.dmap

        return pn.Row(plot, self.widgets)


class SegmentationExportDashBoard(SegmentationDashBoard, RoiExportCallback):
    @param.depends('_complete_update_counter')
    def dmap(self):
        dmaps = self._get_channel_dmaps()
        dmaps.append(self.roi_plot)
        dmap = self.segmentation_viewer(dmaps)

        self.export_viewers = self.segmentation_viewer.channel_viewers

        return dmap

    @param.depends('_complete_update_counter')
    def widgets(self):
        wg = [super().widgets(), self._export_widgets()]
        return pn.Column(*wg)


# TODO
# - test on images of the same size
# - test with spacing != 1
# - fix + test 3D img
# - 3D drawing thickness
# - demo/test notebook


class AnnotationDashBoard(SegmentationDashBoard):

    annot_channel = param.Parameter(doc='id of annotation channel')
    freehand_editor = param.Parameter(FreehandEditor())
    old_subdf = param.DataFrame()

    def _make_dataset(self, key, img):
        if key == self.annot_channel:
            annot_dataset = EditableHvDataset(img=img,
                                              label=self.index_to_str(key))
            # force reset drawing tool axis
            self.freehand_editor = FreehandEditor(dataset=annot_dataset)
            return annot_dataset
        else:
            return HvDataset(img=img, label=self.index_to_str(key))

    # NOTE overriding base class --> watch=True not needed (else triggers double update)
    @param.depends('_dynamic_update_counter')
    def _dynamic_img_update(self):
        self.save_annot()

        for hv_ds, img in zip(self.hv_datasets, self.loaded_objects.values()):
            hv_ds.img = img

    @param.depends('_complete_update_counter')
    def dmap(self):
        self.save_annot()

        if not self.segmentation_viewer.channel_viewers or self._has_multiselect_changed:
            selected_channel_config = {
                key: self.channel_config[key]
                for key in self.loaded_objects.keys()
            }
            self.segmentation_viewer = SegmentationViewer.from_channel_config(
                selected_channel_config,
                composite_channels=self.composite_channels,
                overlay_channels=self.overlay_channels)
            self._widget_update_counter += 1

        self.hv_datasets = [
            self._make_dataset(key, img)
            for key, img in self.loaded_objects.items()
        ]
        dmaps = [hv_ds.dmap() for hv_ds in self.hv_datasets]

        # apply slicer if 3d image
        if next(iter(self.loaded_objects.values())).ndim > 2:
            dmaps = [self.slicer(dmap) for dmap in dmaps]

        # NOTE: workaround to overlay drawingtool. does not work if overlayed after Overlay + collate
        # similar to reported holoviews bug. tap stream attached to a dynamic map does not update
        # https://github.com/holoviz/holoviews/issues/3533
        dmaps.append(self.freehand_editor.path_plot)
        dmap = self.segmentation_viewer(dmaps)

        # Note
        # dmap * self.freehand_editor.path_plot does not work (no drawing tool available)
        # self.freehand_editor.path_plot * dmap works but path is drawn behind the image
        return dmap

    def save_annot(self, event=None):
        npimg = self.freehand_editor.dataset.img.astype(np.int16)
        if npimg.shape != (2, 2) and self.old_subdf is not None:
            single_index = list(
                set(self.old_subdf.index.names) -
                set(self.multi_select_levels))
            row = self.old_subdf.reset_index(single_index).dc[
                self.annot_channel]
            row.dc.write(npimg, compress=9)

    def discard_changes(self, event=None):

        single_index = list(
            set(self.old_subdf.index.names) - set(self.multi_select_levels))
        row = self.old_subdf.reset_index(single_index).dc[self.annot_channel]
        img = row.dc.read()[0]

        self.freehand_editor.dataset.img = img

    @param.depends('subdf', watch=True)
    def _backup_subdf(self):
        self.old_subdf = self.subdf

    def widgets(self):
        wg = super().widgets()

        save_button = pn.widgets.Button(name='save')
        save_button.on_click(self.save_annot)

        discard_button = pn.widgets.Button(name='discard changes')
        discard_button.on_click(self.discard_changes)

        edit_wg = self.freehand_editor.widgets()
        edit_wg.append(save_button)
        edit_wg.append(discard_button)

        return pn.Column(wg, edit_wg)


class OrthoSegmentationDashBoard(BaseImageDashBoard):
    '''Dashboard to views 2D, multi-channel images as color composite.'''

    channel_config = param.Dict({}, doc='dictionnary configuring each channel')
    composite_channels = param.List(
        doc='ids of channels to be displayed as color composite')
    overlay_channels = param.List(
        doc='ids of channels to be displayed as overlay on top of the composite'
    )
    segmentation_viewer = param.Parameter(SegmentationViewer())
    hv_datasets = param.List()
    ortho_viewer = param.Parameter(OrthoViewer(add_crosshairs=False))
    spacing = param.Parameter((1, ), doc='pixel/voxel size', precedence=-1)
    init_position = param.Array(np.array([-1, -1, -1]))

    last_clicked_position = param.Array(np.array([]))

    _widget_update_counter = param.Integer(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @param.depends('ortho_viewer.z_viewer.slice_id',
                   'ortho_viewer.y_viewer.slice_id',
                   'ortho_viewer.x_viewer.slice_id',
                   watch=True)
    def watch_position(self):
        z = self.ortho_viewer.z_viewer.slice_id
        y = self.ortho_viewer.y_viewer.slice_id
        x = self.ortho_viewer.x_viewer.slice_id

        pos = np.array(np.array((z, y, x)) / self.spacing)
        self.last_clicked_position = np.round(pos).astype(int)

    @param.depends('_dynamic_update_counter', watch=True)
    def _dynamic_img_update(self):
        for hv_ds, img in zip(self.hv_datasets, self.loaded_objects.values()):
            hv_ds.img = img

    def dmap(self):

        if not self.segmentation_viewer.channel_viewers or self._has_multiselect_changed:
            selected_channel_config = {
                key: self.channel_config[key]
                for key in self.loaded_objects.keys()
            }
            self.segmentation_viewer = SegmentationViewer.from_channel_config(
                selected_channel_config,
                composite_channels=self.composite_channels,
                overlay_channels=self.overlay_channels)
            self._widget_update_counter += 1

        self.hv_datasets = [
            HvDataset(img=img,
                      label=self.index_to_str(key),
                      spacing=self.spacing)
            for key, img in self.loaded_objects.items()
        ]
        dmaps = [hv_ds.dmap() for hv_ds in self.hv_datasets]

        dmaps = [self.ortho_viewer(dmap) for dmap in dmaps]

        # invert slices and channels
        dmaps = list(zip(*dmaps))

        # add crosshair overlay, bug if adding to an existing overlay
        cross = self.ortho_viewer.get_crosshair()
        dmaps = [dmap + cr for dmap, cr in zip(dmaps, cross)]

        dmaps = [self.segmentation_viewer(dmap) for dmap in dmaps]

        return dmaps

    @param.depends('_widget_update_counter')
    def widgets(self):

        wg = [super().widgets(), self.segmentation_viewer.widgets]
        wg = list(filter(None, wg))
        return pn.Column(*wg)

    @param.depends('_complete_update_counter')
    def _rebuild_panel(self):
        self.ortho_viewer = OrthoViewer(add_crosshairs=False,
                                        target_position=self.init_position)

        panel = self.ortho_viewer.panel(self.dmap())

        # add the composite viewer above the orthoview widget (navigation checkbox)
        panel[1][1] = pn.Column(self.widgets(), panel[1][1])

        return panel

    def panel(self):
        return pn.Row(self._rebuild_panel)


class ScatterDashBoard(MultiCollectionHandler, param.Parameterized):
    '''Dashboard to view 3 dimensions (x,y,color) of a multidimensional dataset 
    as a scatter plot.'''

    x_key = param.Selector()
    y_key = param.Selector()
    color_key = param.Selector()
    hover_keys = param.List()
    filter_columns = param.List()

    selected_row = param.Series(pd.Series([], dtype=int))
    selection_ids = param.List([])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        available_keys = self.df.columns.tolist()
        self.param.x_key.objects = available_keys
        self.param.y_key.objects = available_keys
        self.param.color_key.objects = available_keys

        if not self.x_key:
            self.x_key = available_keys[0]

        if not self.y_key:
            self.y_key = available_keys[1]

        if not self.color_key:
            self.color_key = available_keys[2]

        if isinstance(self.df.index, pd.MultiIndex):
            self.multi_select_levels = list(self.df.index.names)
        else:
            self.multi_select_levels = [self.df.index.name]

        self.file_widgets = []
        self._build_widgets()

    def _selected_hook(self, plot, element):
        '''Directly access bokeh figure and set selection'''
        # NOTE only works when called during plot creation
        # --> redraw entire figure to update selection
        # In principle it's possible, from a handle, to update only
        # the selection but update is not triggered

        plot.handles['selected'].indices = self.selection_ids

    @param.depends('x_key', 'y_key', 'color_key', 'selection_ids', 'subdf')
    def plot_scatter(self):
        points = hv.Points(self.subdf,
                           kdims=[self.x_key, self.y_key],
                           vdims=self.hover_keys + [self.color_key],
                           group='props_scatter')

        # change colormap for categorical values
        if self.subdf[self.color_key].dtype == 'O':
            if len(self.subdf[self.color_key].unique()) <= 10:
                cmap = 'Category10'
            else:
                cmap = 'Category20'
            colorbar = False
            color_levels = None
        else:
            cmap = 'viridis'
            colorbar = True
            color_levels = len(self.subdf[self.color_key].unique())

        points.opts(
            color=self.color_key,
            color_levels=color_levels,
            cmap=cmap,
            colorbar=colorbar,
            tools=['hover', 'box_select', 'lasso_select', 'tap'],
        )

        # add selection stream and attach callback to update sample/image selection
        self._selection = Selection1D(source=points)

        @param.depends(self._selection.param.index, watch=True)
        def update_image_selectors(index):
            if len(index) > 0:
                self.selected_row = self.subdf.iloc[index[0]]

        points.opts(hooks=[self._selected_hook])

        # TODO fix broken adjoint histograms, broken in latest holviews when points color is set
        return points  #.hist(dimension=[self.x_key, self.y_key], num_bins=100).opts(opts.Histogram(color='grey', yaxis='bare', xaxis='bare'))

    def widget(self):
        scatter_wg = pn.WidgetBox(
            'Scatter',
            self.param.x_key,
            self.param.y_key,
            self.param.color_key,
        )

        return pn.Row(scatter_wg, super().widgets)

    def panel(self):
        return pn.Column(self.plot_scatter, self.widget())
