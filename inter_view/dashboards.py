import numpy as np
import pandas as pd
import param
import panel as pn
import holoviews as hv

from holoviews import opts, dim
from holoviews.streams import Selection1D
from bokeh.models import HoverTool

from inter_view.utils import label_cmap, split_element, rasterize_custom, zoom_bounds_hook
from inter_view.view_images import SliceViewer, OverlayViewer, OrthoViewer
from inter_view.edit_images import LabelEditor, FreehandEditor
from inter_view.io import ImageHandler

# TODO rename group according to plot (slice, orthoview, etc.) --> set option accordign to names

opts.defaults(
    opts.Points(
        'props_scatter',
        frame_width=600,
        frame_height=600,
        alpha=0.5,
        size=5,
        line_color=None,
        nonselection_fill_alpha=0.3,
        selection_line_color='black',
        selection_line_alpha=1.,
        active_tools=['wheel_zoom'],
        colorbar_position='left',
        title='',
    ))


class SegmentationSliceDashBoard(param.Parameterized):
    '''Dashboard to view segmentation results (raw + segm channels)'''

    image_handler = param.Parameter()
    slice_viewer = param.Parameter()
    overlay_viewer = param.Parameter()
    plot_width = param.Integer(500)

    include_background = param.Boolean(
        False,
        doc=
        """Whether the background should be considered as a normal label or transparent"""
    )

    def __init__(self, df, ch_col, raw, segm, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.slice_viewer = SliceViewer()
        self.overlay_viewer = OverlayViewer()

        self.raw = raw
        self.segm = segm
        self.ch_col = ch_col

        # keep rows with channels of interest
        df = df[df.index.get_level_values(ch_col).isin([raw, segm])]

        # keeps row only if raw-segm pair is available
        df = pd.concat([
            g
            for _, g in df.groupby([n for n in df.index.names if n != ch_col])
            if len(g) == 2
        ])

        self.image_handler = ImageHandler(df, ds_dims=[ch_col])

    @param.depends('image_handler.load_count')
    def plot_image(self):
        self.slice_viewer.reset()

        dmaps = split_element(self.image_handler.ds,
                              self.ch_col,
                              values=[self.raw, self.segm])

        if len(dmaps[0].kdims) > 2:
            dmaps = tuple(self.slice_viewer(dmap) for dmap in dmaps)
        else:
            dmaps = tuple(
                hv.util.Dynamic(dmap, operation=lambda x: x.to(hv.Image)).
                relabel(dmap.label) for dmap in dmaps)

        dmaps = tuple(rasterize_custom(dmap, [self.segm]) for dmap in dmaps)
        dmaps = self.overlay_viewer(dmaps)

        hover = HoverTool(tooltips=[('label id', '@image')])
        dmaps = dmaps.opts(
            opts.Overlay(normalize=False),
            opts.Image(data_aspect=1.,
                       show_legend=False,
                       normalize=False,
                       xaxis='bare',
                       yaxis='bare',
                       cmap='greys_r',
                       frame_width=self.plot_width),
            opts.Overlay(show_title=False),
            opts.Image(self.segm,
                       cmap=label_cmap,
                       clipping_colors={'min': (0, 0, 0, 0)},
                       clim=(int(not self.include_background),
                             len(label_cmap)),
                       tools=[hover]),
        )

        return dmaps

    def panel(self):

        # evaluate image first, then alpha slider
        hvimg = pn.panel(self.plot_image)
        alphas_wg = pn.panel(self.overlay_viewer.widget)

        # add z slider if 3D
        if len(self.slice_viewer.slicer.param.slice_id.objects) > 1:
            return pn.Row(
                hvimg,
                pn.Column(self.image_handler.widgetbox,
                          self.slice_viewer.slicer.widget,
                          alphas_wg)).servable()
        else:
            return pn.Row(hvimg,
                          pn.Column(self.image_handler.widgetbox,
                                    alphas_wg)).servable()


class SegmentationOrthoDashBoard(param.Parameterized):
    '''Dashboard to view segmentation results (raw + segm channels)'''

    image_handler = param.Parameter()
    ortho_viewer = param.Parameter()
    overlay_viewer = param.Parameter()
    plot_width = param.Integer(500)

    include_background = param.Boolean(
        False,
        doc=
        """Whether the background should be considered as a normal label or transparent"""
    )

    def __init__(self,
                 df,
                 ch_col,
                 raw,
                 segm,
                 spacing_col=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.ortho_viewer = OrthoViewer(ref_width=self.plot_width)
        self.overlay_viewer = OverlayViewer()

        self.raw = raw
        self.segm = segm
        self.ch_col = ch_col

        # keep rows with channels of interest
        df = df[df.index.get_level_values(ch_col).isin([raw, segm])]

        # keeps row only if raw-segm pair is available
        df = pd.concat([
            g
            for _, g in df.groupby([n for n in df.index.names if n != ch_col])
            if len(g) == 2
        ])

        self.image_handler = ImageHandler(df,
                                          ds_dims=[ch_col],
                                          spacing_col=spacing_col)

        hover = HoverTool(tooltips=[('label id', '@image')])

        opts.defaults(
            opts.Overlay(normalize=False),
            opts.Image('Image.{}'.format(raw),
                       data_aspect=1.,
                       cmap='greys_r',
                       show_legend=False,
                       bgcolor='black'),
            opts.HLine(line_dash='dashed', line_width=2, line_color='white'),
            opts.VLine(line_dash='dashed', line_width=2, line_color='white'),
            opts.Image('Image.{}'.format(segm),
                       data_aspect=1.,
                       cmap=label_cmap,
                       show_legend=False,
                       clipping_colors={'min': (0, 0, 0, 0)},
                       clim=(int(not self.include_background),
                             len(label_cmap)),
                       tools=[hover],
                       bgcolor='black'))

    @param.depends('image_handler.load_count')
    def plot_orthoview(self):
        self.ortho_viewer.reset()

        dmaps = split_element(self.image_handler.ds,
                              self.ch_col,
                              values=[self.raw, self.segm])

        ortho_dmaps = tuple(self.ortho_viewer(dmap) for dmap in dmaps)
        ortho_dmaps = tuple(
            tuple(rasterize_custom(dmap, [self.segm]) for dmap in dmaps)
            for dmaps in ortho_dmaps)

        # swap orthoview and overlay dimensions
        ortho_dmaps = tuple(zip(*ortho_dmaps))

        ortho_dmaps = tuple(
            self.overlay_viewer(dmaps) for dmaps in ortho_dmaps)

        l = self.ortho_viewer.panel(*ortho_dmaps)
        l[1][1] = pn.Column(self.image_handler.widgetbox,
                            self.overlay_viewer.widget)

        return l

    def panel(self):
        return pn.panel(self.plot_orthoview).servable()


class AnnotationDashBoard(param.Parameterized):
    '''Dashboard to view segmentation results (raw + segm channels)'''

    image_handler = param.Parameter()
    slice_viewer = param.Parameter()
    overlay_viewer = param.Parameter()
    label_editor = param.Parameter()
    freehand_editor = param.Parameter()
    loaded_subdc = param.DataFrame()

    include_background = param.Boolean(
        False,
        doc=
        """Whether the background should be considered as a normal label or transparent"""
    )
    cmap = param.Parameter(label_cmap, precedence=-1)
    discarding = param.Boolean(False)

    plot_width = param.Integer(500)

    def __init__(self,
                 df,
                 ch_col,
                 raw,
                 segm,
                 spacing_col=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.slice_viewer = SliceViewer()
        self.overlay_viewer = OverlayViewer()
        self.label_editor = LabelEditor()
        self.freehand_editor = FreehandEditor(label_editor=self.label_editor,
                                              plot_width=self.plot_width)

        self.raw = raw
        self.segm = segm
        self.ch_col = ch_col

        # keep rows with channels of interest
        df = df[df.index.get_level_values(ch_col).isin([raw, segm])]

        # keeps row only if raw-segm pair is available
        df = pd.concat([
            g
            for _, g in df.groupby([n for n in df.index.names if n != ch_col])
            if len(g) == 2
        ])

        self.image_handler = ImageHandler(df,
                                          ds_dims=[ch_col],
                                          spacing_col=spacing_col)

        if not self.include_background:
            self.cmap = self.cmap[1:]

    @param.depends('image_handler.load_count')
    def plot_image(self):
        self.freehand_editor.reset()

        # save previous image
        if self.loaded_subdc is not None and not self.discarding:
            self.save_annot()

        self.discarding = False

        # don't reset the slider if reloading the same image
        if self.loaded_subdc is None or not self.loaded_subdc.equals(
                self.image_handler.subdc):
            self.slice_viewer.reset()

        self.loaded_subdc = self.image_handler.subdc

        raw_ds, segm_ds = split_element(self.image_handler.ds,
                                        self.ch_col,
                                        values=[self.raw, self.segm])

        # extract data array in label editor
        segm_dmap = self.label_editor(segm_ds)
        #add drawing tool
        segm_dmap = self.freehand_editor(
            segm_dmap,
            slicing_info=lambda: (0, self.slice_viewer.slicer.slice_id))
        #         dmaps = self.slice_viewer(segm_dmap) * self.freehand_editor.drawingtool

        if len(segm_ds.kdims) > 2:
            dmaps = tuple(
                self.slice_viewer(dmap) for dmap in (raw_ds, segm_dmap))
        else:
            dmaps = tuple(
                hv.util.Dynamic(dmap, operation=lambda x: x.to(hv.Image))
                for dmap in (raw_ds, segm_dmap))

        dmaps = tuple(rasterize_custom(dmap, [self.segm]) for dmap in dmaps)
        dmaps = self.overlay_viewer(dmaps)

        dmaps = dmaps * self.freehand_editor.drawingtool

        dmaps = dmaps.opts(
            opts.Image(data_aspect=1,
                       show_legend=False,
                       normalize=False,
                       xaxis='bare',
                       yaxis='bare',
                       cmap='greys_r',
                       frame_width=self.plot_width),
            opts.Overlay(show_title=False, normalize=False),
            opts.Image(self.segm,
                       cmap=self.cmap,
                       clipping_colors={'min': (0, 0, 0, 0)},
                       clim=(int(not self.include_background),
                             len(self.cmap))),
        )

        # prevent from zooming outside of image
        # TODO automatically adjust as function of slicing axis
        xaxis_vals = raw_ds.dimension_values('x')
        yaxis_vals = raw_ds.dimension_values('y')
        bounds = (xaxis_vals.min(), yaxis_vals.min(), xaxis_vals.max(),
                  yaxis_vals.max())

        dmaps = dmaps.opts(hooks=[zoom_bounds_hook(bounds)])

        return dmaps

    def save_annot(self, event=None):
        npimg = self.label_editor.array.astype(np.int16)
        print('saving {}'.format(
            self.loaded_subdc.set_index(
                self.ch_col).lsc[self.segm].lsc.path[0]))
        self.loaded_subdc.set_index(self.ch_col).lsc[self.segm].lsc.write(
            npimg,
            compressed=False)  # compression does not handle negative label

    def discard_changes(self, event=None):
        self.discarding = True
        self.image_handler.update_ds()

    def panel(self):

        save_button = pn.widgets.Button(name='save')
        save_button.on_click(self.save_annot)

        discard_button = pn.widgets.Button(name='discard changes')
        discard_button.on_click(self.discard_changes)

        # evaluate image first, then alpha slider
        hvimg = pn.panel(self.plot_image)
        alphas_wg = pn.panel(self.overlay_viewer.widget)

        ih_widgets = self.image_handler.widgetbox
        ih_widgets.append(save_button)
        ih_widgets.append(discard_button)

        # add z slider if 3D
        if len(self.slice_viewer.slicer.param.slice_id.objects) > 1:
            return pn.Row(
                pn.Column(hvimg, self.slice_viewer.slicer.widget),
                pn.Column(ih_widgets, alphas_wg, self.label_editor.widgets,
                          self.freehand_editor.widgets)).servable()
        else:
            return pn.Row(
                pn.Column(hvimg),
                pn.Column(ih_widgets, alphas_wg, self.label_editor.widgets,
                          self.freehand_editor.widgets)).servable()


class ScatterDashBoard(param.Parameterized):
    '''Dashboard to view 3 dimensions (x,y,color) of a multidimensional dataset 
    as a scatter plot.'''

    x_key = param.Selector()
    y_key = param.Selector()
    color_key = param.Selector()

    props = param.DataFrame(pd.DataFrame())

    selected_row = param.Series(pd.Series([], dtype=int))
    selection_ids = param.List([])

    def __init__(self,
                 props,
                 x_key=None,
                 y_key=None,
                 color_key=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.props = props

        available_keys = props.columns.tolist()
        self.param.x_key.objects = available_keys
        self.param.y_key.objects = available_keys
        self.param.color_key.objects = available_keys

        if x_key:
            self.x_key = x_key
        else:
            self.x_key = available_keys[0]

        if y_key:
            self.y_key = y_key
        else:
            self.y_key = available_keys[1]

        if color_key:
            self.color_key = color_key
        else:
            self.color_key = available_keys[2]

    def _selected_hook(self, plot, element):
        '''Directly access bokeh figure and set selection'''
        # NOTE only works when called during plot creation
        # --> redraw entire figure to update selection
        # In principle it's possible, from a handle, to update only
        # the selection but update is not triggered

        plot.handles['selected'].indices = self.selection_ids

    @param.depends('x_key', 'y_key', 'color_key', 'selection_ids')
    def plot_scatter(self):

        tooltips = [
            ('bloc id', '@bloc_id'),
            ('cell type', '@cell_type'),
            ('nucleus id', '@nucleus_id'),
        ]
        hover = HoverTool(tooltips=tooltips)

        points = hv.Points(self.props,
                           kdims=[self.x_key, self.y_key],
                           group='props_scatter')

        # change colormap for categorical values
        if self.props[self.color_key].dtype == 'O':
            if len(self.props[self.color_key].unique()) <= 10:
                cmap = 'Category10'
            else:
                cmap = 'Category20'
            colorbar = False
            color_levels = None
        else:
            cmap = 'viridis'
            colorbar = True
            color_levels = len(self.props[self.color_key].unique())

        points.opts(
            color=self.color_key,
            color_levels=color_levels,
            cmap=cmap,
            colorbar=colorbar,
            tools=[hover, 'box_select', 'lasso_select', 'tap'],
        )

        # add selection stream and attach callback to update sample/image selection
        self._selection = Selection1D(source=points)

        @param.depends(self._selection.param.index, watch=True)
        def update_image_selectors(index):
            if len(index) > 0:
                self.selected_row = self.props.iloc[index[0]]

        points.opts(hooks=[self._selected_hook])

        # TODO fix broken adjoint histograms, broken in latest holviews when points color is set
        return points  #.hist(dimension=[self.x_key, self.y_key], num_bins=100).opts(opts.Histogram(color='grey', yaxis='bare', xaxis='bare'))

    def widget(self):
        return pn.WidgetBox(
            'Scatter',
            self.param.x_key,
            self.param.y_key,
            self.param.color_key,
        )

    def panel(self):
        return pn.Column(self.plot_scatter, self.widget())
