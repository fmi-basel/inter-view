import numpy as np

import holoviews as hv
hv.extension('bokeh', logo=False)

from holoviews import opts

import param
import panel as pn

from inter_view.utils import Alpha, Slice, split_element, format_as_rgb, flip_axis, rasterize_overlay


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

    def __call__(self, ds):
        dmap = hv.util.Dynamic(self.slicer(ds),
                               operation=self._dynamic_call,
                               shared_data=False)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.z_viewer = SliceViewer(axis='z')
        self.x_viewer = SliceViewer(axis='x')
        self.y_viewer = SliceViewer(axis='y')

    def __call__(self, ds):

        img_xy = self.z_viewer(ds).opts(
            opts.Image(xaxis='bare', yaxis='bare', frame_width=self.ref_width))
        ref_height = int(
            round(self.ref_width * ds.dimension_values('y').max() /
                  ds.dimension_values('x').max()))
        img_zy = self.x_viewer(ds).opts(
            opts.Image(xaxis='bare',
                       yaxis='bare',
                       frame_height=ref_height,
                       invert_axes=True))
        img_xz = self.y_viewer(ds).opts(
            opts.Image(xaxis='bare', yaxis='bare', frame_width=self.ref_width)
        )  #, invert_yaxis=True)).opts(shared_axes=False)
        img_xz = flip_axis(img_xz, axis='z')

        if self.return_panel:
            return self.panel(img_xy, img_zy, img_xz)
        else:
            return img_xy, img_zy, img_xz

    def _find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _init_tap_navigator(self):
        '''move to click position by changing sliders value'''
        @param.depends(x1=self.z_viewer.tap.param.x,
                       x2=self.y_viewer.tap.param.x,
                       watch=True)
        def update_x_slider(x1=None, x2=None):
            x = x1 or x2
            if self.navigaton_on and x is not None:
                x_vals = self.x_viewer.slicer.param.slice_id.objects
                self.x_viewer.slicer.slice_id = self._find_nearest(x_vals, x)

        @param.depends(y1=self.x_viewer.tap.param.y,
                       y2=self.z_viewer.tap.param.y,
                       watch=True)
        def update_y_slider(y1=None, y2=None):
            y = y1 or y2
            if self.navigaton_on and y is not None:
                y_vals = self.y_viewer.slicer.param.slice_id.objects
                self.y_viewer.slicer.slice_id = self._find_nearest(y_vals, y)

        @param.depends(z1=self.y_viewer.tap.param.y,
                       z2=self.x_viewer.tap.param.x,
                       watch=True)
        def update_z_slider(z1=None, z2=None):

            # special case, y axis of xz panel needs to inverted
            if self.navigaton_on and z1 is not None:
                z_vals = self.z_viewer.slicer.param.slice_id.objects
                z = (max(z_vals) - z1)
                self.z_viewer.slicer.slice_id = self._find_nearest(z_vals, z)

            elif self.navigaton_on and z2 is not None:
                z = z2
                z_vals = self.z_viewer.slicer.param.slice_id.objects
                self.z_viewer.slicer.slice_id = self._find_nearest(z_vals, z)

        update_x_slider()
        update_y_slider()
        update_z_slider()

    def panel(self, img_xy, img_zy, img_xz):

        self._init_tap_navigator()

        panel_xy = self.z_viewer.panel(
            (img_xy * hv.DynamicMap(self._xy_crosshair)).opts(title=''))
        panel_zy = self.x_viewer.panel(
            (img_zy * hv.DynamicMap(self._zy_crosshair)).opts(title=''))
        panel_xz = self.y_viewer.panel(
            (img_xz * hv.DynamicMap(self._xz_crosshair)).opts(title=''))

        # ~ panel_xy = self.z_viewer.panel( (img_xy ).opts(title='') )
        # ~ panel_zy = self.x_viewer.panel( (img_zy ).opts(title='') )
        # ~ panel_xz = self.y_viewer.panel( (img_xz ).opts(title='') )

        return pn.Column(pn.Row(panel_xy, panel_zy),
                         pn.Row(panel_xz, self.param.navigaton_on))

    @param.depends('x_viewer.slicer.slice_id', 'y_viewer.slicer.slice_id')
    def _xy_crosshair(self):
        return hv.VLine(self.x_viewer.slicer.slice_id) * hv.HLine(
            self.y_viewer.slicer.slice_id)

    @param.depends('z_viewer.slicer.slice_id', 'y_viewer.slicer.slice_id')
    def _zy_crosshair(self):
        return (hv.HLine(self.z_viewer.slicer.slice_id,
                         kdims=['y', 'z']).redim.label(x='z') *
                hv.VLine(self.y_viewer.slicer.slice_id, kdims=['y', 'z']))
        # ~ return (hv.HLine(self.z_viewer.slicer.slice_id).redim.label(x='z') * hv.VLine(self.y_viewer.slicer.slice_id))

    @param.depends('x_viewer.slicer.slice_id', 'z_viewer.slicer.slice_id')
    def _xz_crosshair(self):
        max_z_val = self.z_viewer.slicer.param.slice_id.objects.max()
        return (hv.VLine(self.x_viewer.slicer.slice_id, kdims=['x', 'z']) *
                hv.HLine(max_z_val - self.z_viewer.slicer.slice_id,
                         kdims=['x', 'z']).redim.label(y='z'))
        # ~ return (hv.VLine(self.x_viewer.slicer.slice_id) * hv.HLine(max_z_val-self.z_viewer.slicer.slice_id).redim.label(y='z'))


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

    def _dynamic_js_link(self, key):
        def _partial_dynamic_js_link(elem):
            print(elem, key)

            self.alphas[key].jslink(elem, value='glyph.global_alpha')
            #             for e in elem:
            #                 self.alphas[key].jslink(e, value='glyph.global_alpha')
            #                 print('linling')
            return elem

        return _partial_dynamic_js_link

    def __call__(self, dmaps):

        if not isinstance(dmaps, dict):
            raise ValueError(
                'OverlayViewer.__call__ expects a dictionary of dmaps, not: {}'
                .format(type(dmaps)))

        for key in dmaps.keys():
            dmaps[key] = dmaps[key].relabel(group='ImageOverlay')

        linked_dmaps = []
        for key, dmap in dmaps.items():

            # create alpha param/slider if first call
            if not self.alphas.get(key, None):
                self.alphas[key] = pn.widgets.FloatSlider(start=0,
                                                          value=1,
                                                          end=1,
                                                          step=0.01,
                                                          name=key)

            dmap = dmap.relabel(group='ImageOverlay')
            if False:  #isinstance(dmap, hv.DynamicMap):
                dmap = hv.util.Dynamic(dmap,
                                       operation=self._dynamic_js_link(key),
                                       shared_data=False)
            else:
                self.alphas[key].jslink(dmap, value='glyph.global_alpha')

            linked_dmaps.append(dmap)

        dmap = hv.Overlay(linked_dmaps).collate()

        if self.return_panel:
            return self.panel(dmap)
        else:
            return dmap

    def panel(self, dmap):
        return pn.Column(dmap, self.widget())

    def widget(self):
        return pn.WidgetBox('alpha',
                            *[alpha for alpha in self.alphas.values()])
