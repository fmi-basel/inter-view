import numpy as np
import holoviews as hv
import param
import panel as pn

from skimage.exposure import rescale_intensity

from holoviews import opts
from holoviews.operation.datashader import rasterize

from inter_view.color import available_cmaps
from inter_view.utils import LastTap, blend_overlay

# defines default options for all viewers
opts.defaults(
    opts.Image('channel',
               frame_width=600,
               invert_yaxis=True,
               xaxis='bare',
               yaxis='bare',
               bgcolor='black',
               active_tools=['pan', 'wheel_zoom'],
               show_title=False),
    opts.RGB('composite',
             frame_width=600,
             invert_yaxis=True,
             xaxis='bare',
             yaxis='bare',
             bgcolor='black',
             active_tools=['pan', 'wheel_zoom'],
             show_title=False),
    opts.HLine('orthoview',
               line_dash='dashed',
               line_width=1,
               line_color='white'),
    opts.VLine('orthoview',
               line_dash='dashed',
               line_width=1,
               line_color='white'),
    opts.Overlay('orthoview', shared_axes=False, show_title=False),
    opts.Overlay('segmentation', show_title=False),
    opts.Image('segmentation',
               frame_width=600,
               invert_yaxis=True,
               xaxis='bare',
               yaxis='bare',
               bgcolor='black',
               active_tools=['pan', 'wheel_zoom'],
               show_title=False),
)


class BaseViewer(param.Parameterized):
    return_panel = param.Boolean(
        False,
        doc=
        """Returns a dynamic map if False (default) or directly a layout if True"""
    )

    def __call__(self, dmap):

        dmap = self._call(dmap)

        if self.return_panel:
            return self.panel(dmap)
        else:
            return dmap

    def _call(self, dmap):
        pass

    def widgets(self):
        pass

    def panel(self, dmap):
        return pn.Row(dmap, self.widgets())


class ChannelViewer(BaseViewer):

    cmap = param.Selector(list(available_cmaps.keys()), doc='Colormap name')
    # actual colormap, needed to handle named colormaps and instances of colormap
    _cmap = param.Parameter(next(iter(available_cmaps.values())))
    intensity_bounds = param.Range(
        doc=
        'Image clipping bounds. If not specified, imgae (min,max) will be used'
    )
    slider_limits = param.NumericTuple(
        (0, 2**16 - 1), doc='(min,max) values for intensity slider')
    opacity = param.Number(1.,
                           bounds=(0., 1.),
                           step=0.01,
                           doc='Channel opacity',
                           instantiate=True)
    bitdepth = param.Selector(
        default=8,
        objects=[8, 16, 'int8', 'int16'],
        doc=
        'bitdepth of the rasterized image. 16 bits is useful for labels with object above 255'
    )
    raster_aggregator = param.String(
        'default',
        doc=
        'Aggreation method to downsample the image. e.g. use "first" for labels'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_intensity_bounds_limits(self.slider_limits)
        self._watch_selected_cmap()

    @param.depends('cmap', watch=True)
    def _watch_selected_cmap(self):
        self._cmap = available_cmaps[self.cmap]

    def _set_intensity_bounds_limits(self, limits):
        '''Updates the bounds of intensity slider and change current value if out of bound'''

        if self.intensity_bounds and self.intensity_bounds[1] > limits[1]:
            self.intensity_bounds = (self.intensity_bounds[0], limits[1])

        elif self.intensity_bounds and self.intensity_bounds[0] < limits[0]:
            self.intensity_bounds = (limits[0], self.intensity_bounds[1])

        self.param.intensity_bounds.bounds = limits

    @param.depends('intensity_bounds')
    def _update_intensity_bounds(self, elem):

        xs = elem.dimension_values(0, expanded=False)
        ys = elem.dimension_values(1, expanded=False)
        img = elem.dimension_values(2, flat=False)
        dtype = {
            8: np.uint8,
            16: np.uint16,
            'int8': np.int8,
            'int16': np.int16
        }[self.bitdepth]

        bounds = self.intensity_bounds
        if bounds is None:
            bounds = (img.min(), img.max())

        img = rescale_intensity(img, in_range=bounds,
                                out_range=dtype).astype(dtype)

        return elem.clone((xs, ys, img))

    @param.depends()
    def _set_aspect_ratio(self, elem):
        # hack to avoid bug with rasterize + invert_yaxis + aspect='equal'
        # manually set data_aspect=1

        options = elem.opts.get().options
        frame_w = options.get('frame_width', None)
        frame_h = options.get('frame_height', None)

        if frame_w and frame_h:
            # already set
            return elem

        wstart, wstop = elem.dimension_values(0, expanded=False)[[0, -1]]
        hstart, hstop = elem.dimension_values(1, expanded=False)[[0, -1]]

        w = wstop - wstart
        h = hstop - hstart

        if frame_w:
            return elem.opts(
                opts.Image(frame_height=int(round(frame_w / w * h))))
        elif frame_h:
            return elem.opts(
                opts.Image(frame_width=int(round(frame_h / h * w))))
        else:
            return elem

    def _call(self, dmap):
        dmap = dmap.apply(lambda ds: ds.to(hv.Image, group='channel'))
        dmap = rasterize(dmap, aggregator=self.raster_aggregator, expand=False)
        dmap = dmap.apply(self._set_aspect_ratio)
        dmap = dmap.apply(self._update_intensity_bounds)
        return dmap.apply.opts(cmap=self.param._cmap, alpha=self.param.opacity)

    def widgets(self):
        return pn.WidgetBox(self.param.cmap, self.param.intensity_bounds,
                            self.param.opacity)


class OverlayViewer(BaseViewer):

    channel_viewers = param.Dict({})

    def _call(self, dmaps):

        if len(dmaps) != len(self.channel_viewers):
            raise ValueError(
                'the number of dmaps does not match the number of channel viewers: {} vs. {}'
                .format(len(dmaps), len(self.channel_viewers)))

        dmaps = [
            v(dmap) for v, dmap in zip(self.channel_viewers.values(), dmaps)
        ]

        return hv.Overlay(dmaps).collate()

    def widgets(self):
        tabs = pn.Tabs()
        tabs.extend([(str(key), val.widgets())
                     for key, val in self.channel_viewers.items()])
        return tabs

    @classmethod
    def from_channel_config(cls, channel_config, *args, **kwars):

        #         {1:{'label':'marker1','cmap':'cyan','intensity_bounds':(0,4000), 'slider_limits':(0,10000)},
        #          2:{'label':'marker2','cmap':'magenta','intensity_bounds':(1000,6000), 'slider_limits':(0,30000)},
        #          4:{'label':'marker4','cmap':'yellow','intensity_bounds':(0,4000), 'slider_limits':(0,10000)}}

        # get valid channel viewer options
        viewer_options = [
            key for key, val in ChannelViewer.__dict__.items()
            if isinstance(val, param.Parameter)
        ]
        viewers = {}

        for viewer_key, config in channel_config.items():
            config = {
                key: val
                for key, val in config.items() if key in viewer_options
            }
            viewers[viewer_key] = ChannelViewer(**config)

        return cls(channel_viewers=viewers, *args, **kwars)


class CompositeViewer(OverlayViewer):
    def _call(self, dmaps):
        overlay = super()._call(dmaps)
        overlay = overlay.apply(blend_overlay)

        return overlay


class SegmentationViewer(OverlayViewer):
    '''Mix a composite channel for the raw images and individual channels overlay for segmentation'''

    composite_channels = param.List(
        doc='ids of channels to be displayed as color composite')
    overlay_channels = param.List(
        doc='ids of channels to be displayed as overlay on top of the composite'
    )

    def _call(self, dmaps):

        # NOTE: workaround to overlay drawingtool. does not work if overlayed after Overlay + collate
        # similar to reported holoviews bug. tap stream attached to a dynamic map does not update
        # https://github.com/holoviz/holoviews/issues/3533
        extra_overlays = dmaps[len(self.channel_viewers):]

        out_dmaps = []

        if self.composite_channels:
            composite_dmaps = [
                ch_viewer(dmap)
                for (key, ch_viewer
                     ), dmap in zip(self.channel_viewers.items(), dmaps)
                if key in self.composite_channels
            ]
            if len(composite_dmaps) > 0:
                composite_dmap = hv.Overlay(composite_dmaps).collate()
                out_dmaps.append(composite_dmap.apply(blend_overlay))

        if self.overlay_channels:
            out_dmaps.extend([
                ch_viewer(dmap)
                for (key, ch_viewer
                     ), dmap in zip(self.channel_viewers.items(), dmaps)
                if key in self.overlay_channels
            ])

        out_dmaps.extend(extra_overlays)

        return hv.Overlay(out_dmaps).collate().relabel(group='segmentation')


class SliceViewer(BaseViewer):
    '''Slices hv.Dataset along the specified axis.'''

    slice_id = param.ObjectSelector(default=0, objects=[0])
    axis = param.String(default='z')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.param.slice_id.label = self.axis
        self._widget = pn.Param(self.param.slice_id,
                                widgets={
                                    'slice_id': {
                                        'type': pn.widgets.DiscreteSlider,
                                        'formatter': '%.2g',
                                        'sizing_mode': 'stretch_width'
                                    }
                                })[0]

    def update_slider_coords(self, element):
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

    @param.depends('slice_id', 'axis')
    def _slice_volume(self, element):
        if not self.initialized:
            self.update_slider_coords(element)
            self.initialized = True

        new_dims_name = [d.name for d in element.kdims if d.name != self.axis]
        return element.select(**{
            self.axis: self.slice_id
        }).reindex(new_dims_name)

    def _call(self, dmap):
        # reset slider range on first dynamic slice call
        # slider values will corresponds to the last call (intendend for overlay, all dataset should have the same shape)
        self.initialized = False
        return dmap.apply(self._slice_volume)

    def widgets(self):
        return self._widget

    def panel(self, dmap):
        return pn.Column(dmap, self.widgets())


class OrthoViewer(BaseViewer):
    '''Slices a 3D dataset along x,y and z axes and synchronizes the views.'''

    navigaton_on = param.Boolean(True)
    z_viewer = param.Parameter(SliceViewer(axis='z'))
    x_viewer = param.Parameter(SliceViewer(axis='x'))
    y_viewer = param.Parameter(SliceViewer(axis='y'))

    xy_tap = param.Parameter(LastTap())
    zy_tap = param.Parameter(LastTap())
    xz_tap = param.Parameter(LastTap())

    add_crosshairs = param.Boolean(True)

    @param.depends()
    def _invert_axes(self, elem):
        # NOTE should use opts(invert_axes) instead but for some reason
        # it fails after zooming or panning
        return elem.reindex(elem.kdims[::-1])

    def get_crosshair(self):
        self.xy_v = hv.VLine(self.x_viewer._widget.value,
                             kdims=['x', 'y'],
                             label='xyV',
                             group='orthoview')
        self.xy_h = hv.HLine(self.y_viewer._widget.value,
                             kdims=['x', 'y'],
                             label='xyH',
                             group='orthoview')

        self.zy_v = hv.VLine(self.z_viewer._widget.value,
                             kdims=['za', 'y'],
                             label='zyV',
                             group='orthoview')
        self.zy_h = hv.HLine(self.y_viewer._widget.value,
                             kdims=['za', 'y'],
                             label='zyH',
                             group='orthoview')

        self.xz_v = hv.VLine(self.x_viewer._widget.value,
                             kdims=['x', 'zb'],
                             label='xzV',
                             group='orthoview')
        self.xz_h = hv.HLine(self.z_viewer._widget.value,
                             kdims=['x', 'zb'],
                             label='xzH',
                             group='orthoview')

        return [(self.xy_v, self.xy_h), (self.zy_v, self.zy_h),
                (self.xz_v, self.xz_h)]

    def _link_crosshairs(self):

        # move crosshair to slider position (that should now be initizaled)
        self.xy_v.data = self.x_viewer._widget.value
        self.xy_h.data = self.y_viewer._widget.value
        self.zy_v.data = self.z_viewer._widget.value
        self.zy_h.data = self.y_viewer._widget.value
        self.xz_v.data = self.x_viewer._widget.value
        self.xz_h.data = self.z_viewer._widget.value

        self._jslink_discrete_slider(self.x_viewer._widget, self.xy_v)
        self._jslink_discrete_slider(self.y_viewer._widget, self.xy_h)
        self._jslink_discrete_slider(self.z_viewer._widget, self.zy_v)
        self._jslink_discrete_slider(self.y_viewer._widget, self.zy_h)
        self._jslink_discrete_slider(self.x_viewer._widget, self.xz_v)
        self._jslink_discrete_slider(self.z_viewer._widget, self.xz_h)

    def _jslink_discrete_slider(self, widget, line):
        '''hack to jslink pn.widgets.DiscreteSlider to vertical/horizontal lines.
        links the underlying IntSlider and index list of available values'''

        code = '''
                    var vals = {};  
                    glyph.location = vals[source.value]
                '''.format(str(widget.values))

        return widget._slider.jslink(line, code={'value': code})

    @param.depends()
    def _update_dynamic_values(self, xy, zy, xz):
        '''render dummy plots to force updating the sliders, getting plot size, etc.'''
        self.frame_y_size = hv.render(xy).frame_height
        hv.render(zy)  # init slicer
        self.frame_z_size = hv.render(xz).frame_height

    def _call(self, dmap):
        dmap_xy = self.z_viewer(dmap)
        dmap_zy = self.x_viewer(dmap).redim(z='za').apply(self._invert_axes)
        dmap_xz = self.y_viewer(dmap).redim(z='zb')

        self._init_tap_navigator(dmap_xy, dmap_zy, dmap_xz)

        return (dmap_xy, dmap_zy, dmap_xz)

    @param.depends('xy_tap.c0', 'xy_tap.c1', watch=True)
    def _update_xy_sliders(self):
        if self.navigaton_on:
            self.x_viewer.moveto(self.xy_tap.c0)
            self.y_viewer.moveto(self.xy_tap.c1)

    @param.depends('zy_tap.c0', 'zy_tap.c1', watch=True)
    def _update_zy_sliders(self):
        if self.navigaton_on:
            self.z_viewer.moveto(self.zy_tap.c0)
            self.y_viewer.moveto(self.zy_tap.c1)

    @param.depends('xz_tap.c0', 'xz_tap.c1', watch=True)
    def _update_xz_sliders(self):
        if self.navigaton_on:
            self.x_viewer.moveto(self.xz_tap.c0)
            self.z_viewer.moveto(self.xz_tap.c1)

    def _init_tap_navigator(self, xy, zy, xz):
        self.xy_tap(xy)
        self.zy_tap(zy)
        self.xz_tap(xz)

    def panel(self, dmaps):
        xy, zy, xz = dmaps

        self._update_dynamic_values(xy, zy, xz)
        zy.opts(
            opts.Image(frame_width=self.frame_z_size,
                       frame_height=self.frame_y_size),
            opts.RGB(frame_width=self.frame_z_size,
                     frame_height=self.frame_y_size),
        )

        if self.add_crosshairs:
            self.get_crosshair()
            panel_xy = self.z_viewer.panel(
                (xy * self.xy_h * self.xy_v).relabel(group='orthoview'))
            panel_zy = self.x_viewer.panel(
                (zy * self.zy_h * self.zy_v).relabel(group='orthoview'))
            panel_xz = self.y_viewer.panel(
                (xz * self.xz_h * self.xz_v).relabel(group='orthoview'))
        else:
            panel_xy = self.z_viewer.panel(xy.relabel(group='orthoview'))
            panel_zy = self.x_viewer.panel(zy.relabel(group='orthoview'))
            panel_xz = self.y_viewer.panel(xz.relabel(group='orthoview'))

        self._link_crosshairs()

        return pn.Column(pn.Row(panel_xy, panel_zy),
                         pn.Row(panel_xz, self.param.navigaton_on))
