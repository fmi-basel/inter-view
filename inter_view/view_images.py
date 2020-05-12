import numpy as np

import holoviews as hv

from holoviews import opts

import param
import panel as pn

from inter_view.utils import Slice


class SliceViewer(param.Parameterized):
    '''Turns a gridded dataset into hv.Image with a slider for the requested axis'''

    return_panel = param.Boolean(
        False,
        doc=
        """Returns a dynamic map if False (default) or directly a layout if True"""
    )

    def __init__(self, axis='z', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.slicer = Slice(axis=axis)
        self.initialized = False

    def reset(self):
        self.initialized = False
        self.slicer.reset()

    def __call__(self, ds):
        ds = self.slicer(ds)

        # maintain static/dynamic behavior
        if isinstance(ds, hv.DynamicMap):
            dmap = hv.util.Dynamic(ds, operation=self._dynamic_call)
        else:
            dmap = self._dynamic_call(ds)

        if not self.initialized:
            self.initialized = True

            # add tap stream to watch last clicked position
            self.tap = hv.streams.SingleTap(transient=True, source=dmap)

        if self.return_panel:
            return self.panel(dmap)
        else:
            return dmap

    def _dynamic_call(self, ds):
        if len(ds.vdims) == 3:
            return ds.to(hv.RGB)

        return ds.to(hv.Image)

    def panel(self, dmap):
        return pn.Column(dmap, self.slicer.widget)


class OrthoViewer(param.Parameterized):
    return_panel = param.Boolean(
        False,
        doc=
        """Returns a dynamic map if False (default) or directly a layout if True"""
    )
    ref_width = param.Integer(500, doc="""width of xy panel""")
    navigaton_on = param.Boolean(True)
    axiswise = param.Boolean(True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.z_viewer = SliceViewer(axis='z')
        self.x_viewer = SliceViewer(axis='x')
        self.y_viewer = SliceViewer(axis='y')

        self.initialized = False

    def reset(self):
        self.initialized = False
        self.z_viewer.reset()
        self.x_viewer.reset()
        self.y_viewer.reset()

    def __call__(self, ds):
        img_xy = self.z_viewer(ds).opts(axiswise=self.axiswise,
                                        invert_yaxis=True)
        img_zy = self.x_viewer(ds).opts(axiswise=self.axiswise,
                                        invert_yaxis=True,
                                        invert_axes=False)
        img_xz = self.y_viewer(ds).opts(axiswise=self.axiswise,
                                        invert_yaxis=True)

        if not self.initialized:
            self.initialized = True
            self._update_ref_height()
            self._init_crosshairs()

        if self.return_panel:
            return self.panel(img_xy, img_zy, img_xz)
        else:
            return img_xy, img_zy, img_xz

    def _update_ref_height(self):
        y_max = self.y_viewer.slicer.param.slice_id.objects.max()
        x_max = self.x_viewer.slicer.param.slice_id.objects.max()
        self.ref_height = int(round(self.ref_width * y_max / x_max))

    def _init_crosshairs(self):

        # vertical/horizontal orientation prior to switching axis

        self.xy_v = hv.VLine(self.x_viewer.slicer.widget.value,
                             kdims=['x', 'y'],
                             label='xyV').opts(axiswise=self.axiswise)
        self.xy_h = hv.HLine(self.y_viewer.slicer.widget.value,
                             kdims=['x', 'y'],
                             label='xyH').opts(axiswise=self.axiswise)

        self._jslink_discrete_slider(self.x_viewer.slicer.widget, self.xy_v)
        self._jslink_discrete_slider(self.y_viewer.slicer.widget, self.xy_h)

        self.zy_v = hv.VLine(self.z_viewer.slicer.widget.value,
                             kdims=['y', 'z'],
                             label='zyV').opts(axiswise=self.axiswise)
        self.zy_h = hv.HLine(self.y_viewer.slicer.widget.value,
                             kdims=['y', 'z'],
                             label='zyH').opts(axiswise=self.axiswise)

        self._jslink_discrete_slider(self.z_viewer.slicer.widget, self.zy_v)
        self._jslink_discrete_slider(self.y_viewer.slicer.widget, self.zy_h)

        self.xz_v = hv.VLine(self.x_viewer.slicer.widget.value,
                             kdims=['xb', 'zb'],
                             label='xzV').opts(axiswise=self.axiswise)
        self.xz_h = hv.HLine(self.z_viewer.slicer.widget.value,
                             kdims=['xb', 'zb'],
                             label='xzH').opts(axiswise=self.axiswise)

        self._jslink_discrete_slider(self.x_viewer.slicer.widget, self.xz_v)
        self._jslink_discrete_slider(self.z_viewer.slicer.widget, self.xz_h)

    def _jslink_discrete_slider(self, widget, line):
        '''hack to jslink pn.widgets.DiscreteSlider to vertical/horizontal lines.
        links the underlying IntSlider and index list of available values'''

        code = '''
                    var vals = {};  
                    glyph.location = vals[source.value]
                '''.format(str(widget.values))

        return widget._slider.jslink(line, code={'value': code})

    def _init_tap_navigator(self):
        '''move to click position by changing sliders value'''
        @param.depends(x1=self.z_viewer.tap.param.x,
                       x2=self.y_viewer.tap.param.x,
                       watch=True)
        def update_x_slider(x1=None, x2=None):
            x = x1 or x2
            if self.navigaton_on and x is not None:
                self.x_viewer.slicer.moveto(x)
                # ~x_vals = self.x_viewer.slicer.param.slice_id.objects
                # ~self.x_viewer.slicer.slice_id = self._find_nearest_value(x_vals, x)

        @param.depends(y1=self.x_viewer.tap.param.y,
                       y2=self.z_viewer.tap.param.y,
                       watch=True)
        def update_y_slider(y1=None, y2=None):
            y = y1 or y2
            if self.navigaton_on and y is not None:
                self.y_viewer.slicer.moveto(y)
                # ~y_vals = self.y_viewer.slicer.param.slice_id.objects
                # ~self.y_viewer.slicer.slice_id = self._find_nearest_value(y_vals, y)

        @param.depends(z1=self.y_viewer.tap.param.y,
                       z2=self.x_viewer.tap.param.x,
                       watch=True)
        def update_z_slider(z1=None, z2=None):
            z = z1 or z2
            if self.navigaton_on and z is not None:
                self.z_viewer.slicer.moveto(z)
                # ~z_vals = self.z_viewer.slicer.param.slice_id.objects
                # ~self.z_viewer.slicer.slice_id = self._find_nearest_value(z_vals, z)

    def panel(self, img_xy, img_zy, img_xz):

        self._init_tap_navigator()

        xy = (img_xy * self.xy_h * self.xy_v).opts(axiswise=self.axiswise)
        zy = (img_zy * self.zy_h * self.zy_v).opts(axiswise=self.axiswise)
        xz = (img_xz * self.xz_h * self.xz_v).opts(axiswise=self.axiswise)

        #         xy = (img_xy).opts(axiswise=False)
        #         zy = (img_zy).opts(axiswise=False)
        #         xz = (img_xz).opts(axiswise=False)

        xy.opts(
            opts.Image(xaxis='bare', yaxis='bare', frame_width=self.ref_width))
        zy.opts(
            opts.Image(xaxis='bare',
                       yaxis='bare',
                       frame_height=self.ref_height))
        xz.opts(
            opts.Image(xaxis='bare', yaxis='bare', frame_width=self.ref_width))

        # ~xy.opts(opts.Image(frame_width=self.ref_width))
        # ~zy.opts(opts.Image(frame_height=self.ref_height))
        # ~xz.opts(opts.Image(frame_width=self.ref_width))

        panel_xy = self.z_viewer.panel(xy)
        panel_zy = self.x_viewer.panel(zy)
        panel_xz = self.y_viewer.panel(xz)

        return pn.Column(pn.Row(panel_xy, panel_zy),
                         pn.Row(panel_xz, self.param.navigaton_on))


class OverlayViewer(param.Parameterized):
    '''
    
    Note:
    To handle mix overlay of hv.Image and hv.RGB, the input takes a list of dmaps and 
    the result is returned as hv.Overlay (rather than hv.NdOverlay)'''

    return_panel = param.Boolean(
        False,
        doc=
        """Returns a dynamic map if False (default) or directly a layout if True"""
    )
    alphas = param.Dict({})

    def __call__(self, dmaps):

        linked_dmaps = []
        for dmap in dmaps:
            if not self.alphas.get(dmap.label, None):
                self.alphas[dmap.label] = pn.widgets.FloatSlider(
                    start=0, value=1, end=1, step=0.01, name=dmap.label)

            self.alphas[dmap.label].jslink(dmap, value='glyph.global_alpha')

            linked_dmaps.append(dmap)

        dmap = hv.Overlay(linked_dmaps).collate()

        if self.return_panel:
            return self.panel(dmap)
        else:
            return dmap

    def panel(self, dmap):
        return pn.Column(dmap, self.widget())

    @param.depends('alphas')
    def widget(self):
        return pn.WidgetBox('alpha',
                            *[alpha for alpha in self.alphas.values()])


# TODO adjusting image intensity range:
# does not work for rgb --> seems no color mapping --> must reconstruct 32 bits iamge (rgba)

# ~b = pn.widgets.IntRangeSlider(start=0, end=255, value=(0, 255), step=1)

# ~code = '''glyph.color_mapper.high = source.value[1];
# ~glyph.color_mapper.low = source.value[0]'''

# ~b.jslink(hvimg, code={'value': code})

# ~pn.Row(hvimg, b)

# ~# convert holoviews to bokeh
# ~renderer = hv.renderer('bokeh').instance(mode='server')
# ~bokehfig = hv.render(hvimg)
