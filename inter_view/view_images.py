import numpy as np
import os
import cv2 as cv

from inter_view.utils import read_image_size

from skimage.io import imsave, imread

from bokeh.plotting import figure
from bokeh.models import Range1d, ColorBar, WheelZoomTool
from bokeh.models import BoxSelectTool, LassoSelectTool
from bokeh.models import ColumnDataSource, Legend, LegendItem
from bokeh.models.glyphs import ImageURL
from bokeh.models import FreehandDrawTool, PolyEditTool, PolyDrawTool, HoverTool
from bokeh.models import Slider
from bokeh.models import Span
from bokeh.palettes import grey
from bokeh.events import Tap, DoubleTap, Press
from bokeh.models.widgets import CheckboxButtonGroup, Toggle
from bokeh.layouts import gridplot, layout, row, column, widgetbox

# TODO
# consistent naming: plot(p), renderer(r), glyph(g), layout(l), etc.
# set aspect ratio of all fig size (instead of max_d) --> see ImageStackWithSlider
# outliers selection (when hue=outliers) + save_df
# make image tabs into a bokeh composition --> propagate update functions


class TiledImages():
    '''
    Creates a layout with multiple images.
    
    '''

    def __init__(self, image_paths, **kwargs):

        self.image_paths = image_paths
        self.kwargs = kwargs
        self.plot()

    def update_image_url(self, image_paths):

        if len(image_paths) != len(self.image_paths):
            raise ValueError('Trying to update {} images with {} paths'.format(
                len(self.image_paths), len(image_paths)))

        self.image_paths = image_paths
        for key, path in self.image_paths.items():
            self.plot_image(key)

    def plot(self):

        self.figures = {}
        for key, path in self.image_paths.items():
            self.figures[key] = figure(
                active_scroll='wheel_zoom',
                active_drag='pan',
                toolbar_location=None,
                plot_width=200,
                plot_height=200,
                y_axis_location=None,
                x_axis_location=None,
                title=key,
            )

            self.figures[key].select(WheelZoomTool).maintain_focus = False
            self.figures[key].title.text_font_size = '8pt'

            self.figures[key].outline_line_color = None
            self.figures[key].grid.visible = False
            self.figures[key].background_fill_color = None
            self.figures[key].border_fill_color = None

            self.plot_image(key)

        self.p = row(list(self.figures.values()))

    def plot_image(self, key):

        width, height = read_image_size(self.image_paths[key])
        server_img_url = os.path.join(os.path.basename(os.getcwd()),
                                      self.image_paths[key])

        max_d = max(height, width)
        self.figures[key].x_range = Range1d(start=0,
                                            end=max_d,
                                            bounds=(0, max_d))
        self.figures[key].y_range = Range1d(start=max_d,
                                            end=0,
                                            bounds=(0, max_d))

        img_urls = self.figures[key].select(ImageURL)
        # ~ print(type(img_urls))
        if img_urls:  # update existing
            img_urls[0].url = [server_img_url]
            img_urls[0].w = width
            img_urls[0].h = height
            img_urls[1].url = [server_img_url]
            img_urls[1].w = width
            img_urls[1].h = height
        else:
            self.figures[key].image_url(url=[server_img_url],
                                        x=0,
                                        y=0,
                                        w=width,
                                        h=height,
                                        anchor='top_left')


class ImageWithOverlay():
    '''
    Creates a figure with an image and patches overlaid.
    
    '''

    def __init__(self,
                 image_path,
                 source,
                 patch_x,
                 patch_y,
                 center_x,
                 center_y,
                 tooltips_columns=None,
                 **kwargs):

        self.source = source
        self.kwargs = kwargs

        self.patch_x = patch_x
        self.patch_y = patch_y
        self.center_x = center_x
        self.center_y = center_y

        self.width, self.height = read_image_size(image_path)

        self.server_img_url = os.path.join(os.path.basename(os.getcwd()),
                                           image_path)

        max_d = max(self.height, self.width)
        self.x_range = Range1d(start=0, end=max_d, bounds=(0, max_d))
        self.y_range = Range1d(start=max_d, end=0, bounds=(0, max_d))

        self.tooltips_formatting = [
            ("(x,y)", "($x{0.}, $y{0.})"),
        ]
        if tooltips_columns:
            self.tooltips_formatting += [(s.replace('_', ' '), '@' + s)
                                         for s in tooltips_columns]

        self.patch_config = {
            'line_color': 'color',
            'line_alpha': 1.0,
            'fill_alpha': 0.0,
            'line_width': 2,
            'hover_alpha': 0.5,
            'hover_color': 'pink',
            'nonselection_line_color':
            'white',  #color', # bug when using view, wrong color indexing
            'nonselection_line_alpha': 0.2,
            'nonselection_fill_alpha': 0.0,
            'selection_line_color': 'white',  #color',
            'selection_line_alpha': 1.0,
            'selection_fill_alpha': 0.0,
        }

        self.plot()

    def plot(self):

        self.p = figure(
            x_range=self.x_range,
            y_range=self.y_range,
            tools='tap,wheel_zoom,hover,lasso_select,box_select,pan,reset',
            tooltips=self.tooltips_formatting,
            active_scroll='wheel_zoom',
            active_drag='box_select',
            toolbar_location='above',
            plot_width=600,
            y_axis_location=None,
            x_axis_location=None)

        self.p.select(WheelZoomTool).maintain_focus = False
        self.p.select(BoxSelectTool).select_every_mousemove = True
        self.p.select(LassoSelectTool).select_every_mousemove = True

        self.p.grid.visible = False
        self.p.background_fill_color = None
        self.p.border_fill_color = None
        self.p.outline_line_color = None

        self.p.image_url(url=[self.server_img_url],
                         x=0,
                         y=0,
                         w=self.width,
                         h=self.height,
                         anchor='top_left')

        # hack: invisible points to allow lasso selection
        self.scatter = self.p.scatter(x=self.center_y,
                                      y=self.center_x,
                                      size=0,
                                      alpha=0.,
                                      source=self.source,
                                      **self.kwargs)
        self.patches = self.p.patches(xs=self.patch_y,
                                      ys=self.patch_x,
                                      source=self.source,
                                      name='masks',
                                      **self.patch_config,
                                      **self.kwargs)

        self.p.hover.point_policy = 'follow_mouse'
        self.p.hover.names = ['masks']  # only show tooltips when hover patches


# TODO
# add_tap_callback --> use internal 3 separate callback and propagate by calling additional callback
class OrthoView():
    '''create 3 figures with slider to view 3D image stack
    '''

    def __init__(self, config, **kwargs):

        self.top = ImageStackWithSlider(config, axis=0, **kwargs)
        self.right = ImageStackWithSlider(config, axis=2, **kwargs)
        self.front = ImageStackWithSlider(config, axis=1, **kwargs)

        # sync axis and fix aspect ratio
        self.right.p.y_range = self.top.p.y_range
        self.front.p.x_range = self.top.p.x_range

        # setup additional callback when sliders are moved
        self.top.slider.on_change('value', self.update_top_slice)
        self.right.slider.on_change('value', self.update_right_slice)
        self.front.slider.on_change('value', self.update_front_slice)

        # add callback to move slider to clicked
        self.top.p.on_event(Tap, self.top_tap_callback)
        self.right.p.on_event(Tap, self.right_tap_callback)
        self.front.p.on_event(Tap, self.front_tap_callback)

        # add button to enable navigation
        self.button_toggle_nav = Toggle(label='Navigation OFF', active=False)
        self.button_toggle_nav.on_click(self.toggle_navigation)

        # add slices indicator lines
        span_config = {
            'line_dash': 'dashed',
            'line_width': 2,
            'line_color': 'white',
            'line_alpha': 1.0
        }

        self.top_vspan = Span(dimension='height',
                              location=self.right.slice,
                              **span_config)
        self.top_hspan = Span(dimension='width',
                              location=self.front.slice,
                              **span_config)
        self.right_vspan = Span(dimension='height',
                                location=self.top.slice / self.right.xy_ratio,
                                **span_config)
        self.right_hspan = Span(dimension='width',
                                location=self.front.slice,
                                **span_config)
        self.front_vspan = Span(dimension='height',
                                location=self.right.slice,
                                **span_config)
        self.front_hspan = Span(dimension='width',
                                location=self.top.slice / self.right.xy_ratio,
                                **span_config)

        self.top.p.add_layout(self.top_vspan)
        self.top.p.add_layout(self.top_hspan)
        self.right.p.add_layout(self.right_vspan)
        self.right.p.add_layout(self.right_hspan)
        self.front.p.add_layout(self.front_vspan)
        self.front.p.add_layout(self.front_hspan)

        # add callback to keep images in sync
        self.update_in_progress = False
        for channel in config.keys():
            self.top.renderers[channel].data_source.on_change(
                'data', self.update_image(channel, 'top'))
            self.right.renderers[channel].data_source.on_change(
                'data', self.update_image(channel, 'right'))
            self.front.renderers[channel].data_source.on_change(
                'data', self.update_image(channel, 'front'))

        # build layout
        self.plot()

    def add_drawing_tools(self, drawing_channel, external_callback):
        self.drawing_channel = drawing_channel
        self.top.add_drawing_tools(drawing_channel, external_callback)
        self.right.add_drawing_tools(drawing_channel, external_callback)
        self.front.add_drawing_tools(drawing_channel, external_callback)

    def set_drawing_channel(self, drawing_channel):
        self.drawing_channel = drawing_channel
        self.top.set_drawing_channel(drawing_channel)
        self.right.set_drawing_channel(drawing_channel)
        self.front.set_drawing_channel(drawing_channel)

    def set_drawing_color(self, color):
        self.top.set_drawing_color(color)
        self.front.set_drawing_color(color)
        self.right.set_drawing_color(color)

    def get_drawing_color(self):
        return self.top.get_drawing_color()

    def add_tap_callback(self, callback_fct):
        self.top.add_tap_callback(callback_fct)

    def add_image_on_change_callback(self, channel, callback_fct):
        self.top.add_image_on_change_callback(channel, callback_fct)

    def get_zslice(self):
        return self.top.get_zslice()

    def update_image(self, channel, view='top'):
        '''Return callbacks to force update 2 views when one is modified (e.g. force update right and front when top changes)'''

        def propagate_update_image(attr, old, new):
            if not self.update_in_progress:
                self.update_in_progress = True
                for imstack in {'top', 'right', 'front'} - {view}:
                    getattr(self, imstack).force_update(channel)

                self.update_in_progress = False

        return propagate_update_image

    def set_images(self, images):
        self.top.set_images(images)
        self.right.set_images(images)
        self.front.set_images(images)

        # override plot dimension to match main view
        self.right.plot_height = self.top.plot_height
        self.right.plot_width = int(self.right.plot_height * self.right.width /
                                    self.right.height)
        self.right.update_figure_geometry()

        self.front.plot_width = self.top.plot_width
        self.front.plot_height = int(self.front.plot_width *
                                     self.front.height / self.front.width)
        self.front.update_figure_geometry()

    def get_npimage(self, channel):
        return self.top.get_npimage(channel)

    def get_channel_alpha(self, channel):
        return self.top.get_channel_alpha(channel)

    def get_sampling(self):
        return self.top.get_sampling()

    def update_figure_geometry(self):
        self.top.update_figure_geometry()
        self.right.update_figure_geometry()
        self.front.update_figure_geometry()

    def set_toolsize(self, toolsize):
        self.top.set_toolsize(toolsize)
        self.front.set_toolsize(toolsize)
        self.right.set_toolsize(toolsize)

    def get_toolsize(self):
        return self.top.get_toolsize()

    def set_drawing_alpha(self, alpha):
        self.top.set_drawing_alpha(alpha)
        self.front.set_drawing_alpha(alpha)
        self.right.set_drawing_alpha(alpha)

    def set_channel_alpha(self, channel, alpha):
        self.top.set_channel_alpha(channel, alpha)
        self.front.set_channel_alpha(channel, alpha)
        self.right.set_channel_alpha(channel, alpha)

    def toggle_navigation(self, state):
        if state:
            self.button_toggle_nav.label = 'Navigation ON'
        else:
            self.button_toggle_nav.label = 'Navigation OFF'

    def top_tap_callback(self, event):
        '''Moves right and front sliders to position clicked on top view'''
        if self.button_toggle_nav.active:
            self.right.slider.value = int(round(event.x - 0.5))
            self.front.slider.value = int(round(event.y - 0.5))

    def right_tap_callback(self, event):
        '''Moves top and front sliders to position clicked on right view'''
        if self.button_toggle_nav.active:
            self.top.slider.value = int(
                round(event.x * self.right.xy_ratio - 0.5))
            self.front.slider.value = int(round(event.y - 0.5))

    def front_tap_callback(self, event):
        '''Moves top and right sliders to position clicked on front view'''
        if self.button_toggle_nav.active:
            self.right.slider.value = int(round(event.x - 0.5))
            self.top.slider.value = int(
                round(event.y / self.front.xy_ratio - 0.5))

    def update_top_slice(self, attr, old, new):
        '''Update right and front views when top slider is moved'''
        self.right_vspan.location = (new + 0.5) / self.right.xy_ratio
        self.front_hspan.location = (new + 0.5) * self.front.xy_ratio

    def update_right_slice(self, attr, old, new):
        '''Update top and front views when right slider is moved'''
        self.top_vspan.location = new + 0.5
        self.front_vspan.location = new + 0.5

    def update_front_slice(self, attr, old, new):
        '''Update top and right views when front slider is moved'''
        self.top_hspan.location = new + 0.5
        self.right_hspan.location = new + 0.5

    def force_update(self, channel):
        self.top.force_update(channel)

    def plot(self):

        self.controls = widgetbox([self.button_toggle_nav], width=600)
        self.layout = gridplot([[self.top.layout, self.right.layout],
                                [self.front.layout, self.controls]],
                               merge_tools=False)


class ImageStackWithSlider():
    '''
    '''

    def __init__(self, config, sampling=1, axis=0, **kwargs):
        '''Builds a figure with a slider to navigate trough image slices
        '''

        self.axis = axis
        self.axis_mapping = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
        self.drawing_channel = None
        self.update_in_progress = False

        # transpose and arrange images in column datasource (1 slice per column)
        self.npimages = {}  # images as numpy array
        self.data = {}  # images as dict with one item per slice
        for key, val in config.items():

            npimg = val.get('npimage',
                            np.zeros((2, 16, 16), dtype=np.int16) - 1)
            npimg = np.transpose(npimg, self.axis_mapping[axis])
            npimg = npimg[:, ::
                          -1]  # flip x axis to orient image following std convention (top,left) = (0,0)
            self.data[key] = {
                str(z): [npimg[z]]
                for z in range(npimg.shape[0])
            }
            self.npimages[key] = npimg

        self.sampling = np.broadcast_to(np.asarray(sampling), 3)
        self.sampling = self.sampling[self.axis_mapping[axis]]

        self.kwargs = kwargs

        self.compute_figure_geometry(npimg.shape)

        self.slice = 0
        self.x_range = Range1d(start=0, end=self.width, bounds=(0, self.width))
        self.x_range.on_change('start', self.zoom_callback)
        self.y_range = Range1d(start=self.height,
                               end=0,
                               bounds=(0, self.height))

        self.slider = Slider(title='axis({})'.format(axis),
                             start=0,
                             end=self.n_slices - 1,
                             value=self.slice,
                             step=1)
        self.slider.on_change('value', self.update_slice)

        self.tooltips_formatting = [("(x,y)", "($x{0.}, $y{0.})"),
                                    ('value', '@' + str(self.slice))]

        self.plot(config)

    def compute_figure_geometry(self, img_shape):

        self.n_slices, self.height, self.width = img_shape[0:3]
        self.xy_ratio = self.sampling[1] / self.sampling[2]
        if self.xy_ratio < 1:
            self.width /= self.xy_ratio
        else:
            self.height *= self.xy_ratio

        if self.width / self.height > 1:
            self.plot_width = 600
            self.plot_height = int(600 * self.height / self.width)
        else:
            self.plot_width = int(600 * self.width / self.height)
            self.plot_height = 600

    def set_images(self, images):
        '''expects dict with same keys as in init'''

        for key, val in images.items():
            npimg = val
            npimg = np.transpose(npimg, self.axis_mapping[self.axis])
            npimg = npimg[:, ::
                          -1]  # flip x axis to orient image following std convention (top,left) = (0,0)
            self.data[key] = {
                str(z): [npimg[z]]
                for z in range(npimg.shape[0])
            }
            self.npimages[key] = npimg
            self.force_update(key)

        self.compute_figure_geometry(npimg.shape)
        self.update_figure_geometry()

    def add_image_on_change_callback(self, channel, callback_fct):
        self.renderers[channel].data_source.on_change('data', callback_fct)

    def get_npimage(self, channel):
        return self.npimages[channel]

    def add_tap_callback(self, callback_fct):
        self.p.on_event(Tap, callback_fct)

    def force_update(self, channel):
        '''hack to force bokeh update on client side: reassign datasource dict'''
        self.renderers[channel].data_source.data = self.data[channel]

    def update_figure_geometry(self):

        # update figure
        self.update_slice(None, None, self.n_slices // 2)
        self.slider.end = self.n_slices - 1
        self.slider.value = self.slice
        self.x_range.start = 0
        self.x_range.end = self.width
        self.x_range.bounds = (0, self.width)
        self.y_range.start = self.height
        self.y_range.end = 0
        self.y_range.bounds = (0, self.height)
        self.p.plot_width = self.plot_width
        self.p.plot_height = self.plot_height
        # inconsistent names in bokeh: https://github.com/bokeh/bokeh/issues/4830 --> is plot_width deprecated?
        self.p.width = self.plot_width
        self.p.height = self.plot_height

        # update renderers
        for val in self.renderers.values():
            val.glyph.y = self.height
            val.glyph.dw = self.width
            val.glyph.dh = self.height

        # update drawing glyph size
        self.zoom_callback(None, None, None)

    def update_slice(self, attr, old, new):

        self.slice = new

        # bug tooltips field not updating on server side (always point to the initial slice)
        # adding a new tooltip works but how tho remove the old ones???
        # self.tooltips_formatting[1] = ('value', '@' + str(self.slice))
        # for t in self.p.select(HoverTool):
        # t.tooltips = self.tooltips_formatting

        for key, val in self.renderers.items():
            val.glyph.image = str(self.slice)

    def get_sampling(self):
        return self.sampling

    def get_zslice(self):
        return self.slice

    def plot(self, config):

        self.p = figure(
            x_range=self.x_range,
            y_range=self.y_range,
            toolbar_location='above',
            tools='wheel_zoom,pan,reset,tap',  #hover
            # tooltips=self.tooltips_formatting,
            plot_width=self.plot_width,
            plot_height=self.plot_height,
            active_scroll='wheel_zoom',
            y_axis_location=None,
            x_axis_location=None,
        )

        self.p.select(WheelZoomTool).maintain_focus = False
        self.p.outline_line_color = None
        self.p.grid.visible = False
        self.p.background_fill_color = None
        self.p.border_fill_color = None

        self.renderers = {}
        for key, val in config.items():

            self.renderers[key] = self.p.image(
                image=str(self.slice),
                source=ColumnDataSource(self.data[key]),
                x=0,
                y=self.height,
                dw=self.width,
                dh=self.height,
                palette=val.get('palette', grey(256)),
                global_alpha=val.get('alpha', 1.0),
                legend=key,
                name=key)

            self.renderers[key].glyph.color_mapper.low = val.get('map_low', 0)
            self.renderers[key].glyph.color_mapper.high = val.get(
                'map_high', len(val.get('palette', grey(256))))
            self.renderers[key].visible = val.get('visible', True)

        self.adjust_legend(self.p.legend)
        # self.p.hover.point_policy = 'follow_mouse'
        # self.p.hover.names = ['annotations'] # only show tooltips when hover patches

        self.layout = column([self.p, self.slider])  #, sizing_mode='fixed')

    def set_toolsize(self, toolsize):
        self.toolsize = toolsize
        self.zoom_callback(None, None, None)

    def get_toolsize(self):
        return self.toolsize

    def zoom_callback(self, attr, old, new):
        '''Adjusts glyph size corresponding to drawing width with image coord to screen coord ratio'''

        if self.drawing_channel:
            if self.width > self.height:
                self.draw_r.glyph.line_width = self.toolsize * (
                    self.p.plot_width) / (self.p.x_range.end -
                                          self.p.x_range.start)
            else:
                self.draw_r.glyph.line_width = self.toolsize * (
                    self.p.plot_height) / (self.p.y_range.start -
                                           self.p.y_range.end)

    def set_drawing_channel(self, channel):
        self.drawing_channel = channel

    def set_drawing_alpha(self, alpha):
        '''sets alpha of the drawing tool as well as drawing channel'''

        self.draw_r.glyph.line_alpha = alpha
        self.renderers[self.drawing_channel].glyph.global_alpha = alpha

    def set_channel_alpha(self, channel, alpha):
        self.renderers[channel].glyph.global_alpha = alpha
        if channel == self.drawing_channel:
            self.draw_r.glyph.line_alpha = alpha

    def get_drawing_alpha(self):
        return self.renderers[self.drawing_channel].glyph.global_alpha

    def get_channel_alpha(self, channel):
        return self.renderers[channel].glyph.global_alpha

    def set_drawing_color(self, color):

        # value drawn on image data
        self.drawing_color = color

        # change the glyph color as well
        low = self.renderers[self.drawing_channel].glyph.color_mapper.low
        high = self.renderers[self.drawing_channel].glyph.color_mapper.high
        palette = self.renderers[
            self.drawing_channel].glyph.color_mapper.palette
        mapping = palette[round((color - low) / (high - low) * len(palette))]
        if mapping == (0, 0, 0, 0):
            mapping = 'white'  # draw white lines instead of transparent
        self.draw_r.glyph.line_color = mapping

    def get_drawing_color(self):
        return self.drawing_color

    def add_drawing_tools(self, drawing_channel, external_callback):
        self.drawing_channel = drawing_channel
        self.draw_source = ColumnDataSource({'xs': [], 'ys': []})
        # TODO calculate ratio from fig size, zoom level and image resolution
        self.draw_r = self.p.multi_line('xs',
                                        'ys',
                                        source=self.draw_source,
                                        line_cap='round',
                                        line_join='round',
                                        line_width=1,
                                        line_alpha=self.get_drawing_alpha())
        self.set_drawing_color(-1)
        self.set_toolsize(3)
        self.freehand_draw = FreehandDrawTool(renderers=[self.draw_r],
                                              num_objects=1)
        self.p.add_tools(self.freehand_draw)
        # ~ self.p.toolbar.active_tap = self.freehand_draw

        # ~ r2 = self.p.patches('xs', 'ys', source=self.draw_source, legend='corrections_poly', line_width=0 ,fill_alpha=0.5)
        # ~ c1 = self.p.circle(x=[], y=[], color='red', size=5)
        # ~ self.p.add_tools(PolyDrawTool(vertex_renderer=c1, renderers=[r, r2]))
        # ~ self.p.add_tools(PolyEditTool(vertex_renderer=c1, renderers=[r2]))

        self.draw_source.on_change('data', self.draw_callback)
        self.draw_source.on_change('data', external_callback)

    def draw_callback(self, attr, old, new):
        '''burn drawn glyph in annotation image and delete them
        '''

        # TODO adjust coordinates when axis != 0 and xy_ratio != 1
        if new['xs']:  # check that the callback is not triggered by deletion (i.e. itself)
            xs = np.asarray(np.rint(np.asarray(new['xs'][0]) - 0.5), dtype=int)
            ys = np.asarray(np.rint(np.asarray(new['ys'][0]) - 0.5), dtype=int)
            pts = np.stack([xs, ys], axis=1)
            pts = pts.reshape((-1, 1, 2))

            # draw on master image
            npimg = self.npimages[self.drawing_channel]
            # TODO take care of sampling when drawing on right and front views
            cv.polylines(npimg[self.slice, ::-1], [pts], False,
                         (self.drawing_color), self.toolsize, cv.LINE_4)
            # update datasource
            self.force_update(self.drawing_channel)
            # empty drawing source
            self.draw_source.data = {'xs': [], 'ys': []}

    @staticmethod
    def adjust_legend(legend_handle):
        legend_handle.click_policy = 'hide'
        legend_handle.glyph_width = 0
        legend_handle.glyph_height = 15
        legend_handle.label_standoff = 0
        legend_handle.spacing = 0
        legend_handle.padding = 5
        legend_handle.label_text_font_size = '10pt'
        legend_handle.label_text_baseline = 'middle'
        legend_handle.label_height = 15
        legend_handle.label_width: 25

        legend_handle.background_fill_color = "white"
        legend_handle.background_fill_alpha = 0.3
        legend_handle.inactive_fill_color = 'black'
        legend_handle.inactive_fill_alpha = 0.3
