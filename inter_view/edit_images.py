import numpy as np
import cv2 as cv
import param
import panel as pn
import holoviews as hv

from holoviews import streams, opts

from inter_view.utils import label_cmap
from inter_view.utils import HvDataset
from inter_view.color import glasbey_hv_16bit

opts.defaults(
    opts.Rectangles('ROIedit',
                    fill_color=None,
                    line_width=2,
                    line_color='white',
                    line_dash='dashed'))


class RoiEditor(param.Parameterized):
    '''mixin class to add bounding box editor capabilities'''

    roi_plot = param.Parameter(hv.Rectangles([], group='ROIedit'))
    box_edit = param.Parameter(streams.BoxEdit(), instantiate=True)
    spacing = param.NumericTuple((1, 1), doc='2D pixel size')

    def __init__(self, *args, **kwargs):
        num_objects = kwargs.pop('num_objects', 1)
        super().__init__(*args, **kwargs)

        self.box_edit.num_objects = num_objects
        self.box_edit.source = self.roi_plot

    def img_slice(self):
        '''return image slice in px coordinates'''
        if self.box_edit.data is None or not self.box_edit.data['x0']:
            return None

        # repack dict of 4 lists as a list of (x0,x1,y0,y1)
        rois = list(zip(*self.box_edit.data.values()))

        loc = [(slice(max(0, round(y1 / self.spacing[0])),
                      max(0, round(y0 / self.spacing[0]))),
                slice(max(0, round(x0 / self.spacing[1])),
                      max(0, round(x1 / self.spacing[1]))))
               for x0, x1, y0, y1 in rois]
        return loc


class EditableHvDataset(HvDataset):
    '''Extract a data array from a holoviews element and makes it editable'''

    locked_mask = param.Array(
        precedence=-1, doc='''mask of region that should not be updated''')
    drawing_label = param.Selector(default=1, objects=[-1, 0, 1])
    editor_switches = param.ObjectSelector(
        default='pick label', objects=['-', 'pick label', 'fill label'])
    locking_switches = param.ListSelector(default=[],
                                          objects=['background', 'foreground'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_locked_mask()
        self.update_drawing_label_list()

    def click_callback(self, coords):
        if len(coords) != self.img.ndim:
            raise ValueError(
                'Supplied coordinates: {} does not match the image dimensions: {}'
                .format(coords, self.img.ndim))

        coords = tuple(int(round(c)) for c in coords)
        clicked_label = self.img[coords]

        if 'pick label' == self.editor_switches:
            self.drawing_label = clicked_label
        elif 'fill label' == self.editor_switches:
            mask = self.img == clicked_label
            self.write_label(mask)

    @param.depends('img', 'locking_switches', watch=True)
    def update_locked_mask(self):
        mask = np.zeros_like(self.img, dtype=bool)

        if 'background' in self.locking_switches:
            mask[self.img == 0] = True

        if 'foreground' in self.locking_switches:
            mask[self.img > 0] = True

        self.locked_mask = mask

    def write_label(self, mask):

        new_array = self.img
        new_array[mask & (~self.locked_mask)] = self.drawing_label

        # assign new array to trigger updates
        self.img = new_array

    @param.depends('img', watch=True)
    def update_drawing_label_list(self):
        '''List of label to choose from.'''

        max_label = self.img.max()
        # add an extra label to annotate new objects
        unique_labels = list(range(-1, max_label + 2))

        self.param.drawing_label.objects = unique_labels

        if self.drawing_label not in unique_labels:
            self.drawing_label = -1

    def delete_label(self, event=None):
        self.img[self.img == self.drawing_label] = -1
        self.img = self.img

    @param.depends('img')
    def _drawing_label_wg(self):
        return pn.panel(self.param.drawing_label)

    def widgets(self):
        delete_button = pn.widgets.Button(name='delete selected label')
        delete_button.on_click(self.delete_label)

        editor_switches_wg = pn.Param(
            self.param.editor_switches,
            show_name=True,
            name="on click",
            widgets={'editor_switches': {
                'type': pn.widgets.RadioButtonGroup
            }})

        locking_switches_wg = pn.Param(self.param.locking_switches,
                                       show_name=True,
                                       name='lock',
                                       widgets={
                                           'locking_switches': {
                                               'type':
                                               pn.widgets.CheckButtonGroup
                                           }
                                       })

        return pn.WidgetBox(self._drawing_label_wg, editor_switches_wg,
                            locking_switches_wg, delete_button)


class FreehandEditor(param.Parameterized):
    '''Adds a freehand drawing tool that embeds the drawn path in the image/stack'''

    dataset = param.Parameter(EditableHvDataset(), precedence=-1)
    freehand = param.Parameter(streams.FreehandDraw(num_objects=1),
                               precedence=-1)
    pointer_pos = param.Parameter(streams.PointerXY(),
                                  precedence=-1,
                                  instantiate=True)
    clicked_pos = param.Parameter(streams.SingleTap(transient=True),
                                  precedence=-1,
                                  instantiate=True)
    pipe = param.Parameter(streams.Pipe(data=[]),
                           instantiate=True,
                           precedence=-1)
    path_plot = param.Parameter(hv.Path([]), precedence=-1)

    cmap = param.Parameter(glasbey_hv_16bit, precedence=-1)
    zoom_level = param.Number(1.0, precedence=-1)
    tool_width = param.Integer(20, bounds=(1, 300))

    zoom_range = param.Parameter(
        streams.RangeX(),
        doc=
        '''range stream used to adjust glyph size based on zoom level, assumes data_aspect=1''',
        precedence=-1)
    plot_size = param.Parameter(streams.PlotSize(), precedence=-1)
    zoom_level = param.Number(1.0, precedence=-1)
    zoom_initialized = param.Boolean(False, precedence=-1)

    slicer = param.Parameter(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # grey glyph for drawing label -1 (unlabeled)
        self.cmap[-1] = '#999999'

        self.path_plot = hv.DynamicMap(self.plot_path, streams=[self.pipe])
        self.freehand.source = self.path_plot

        self.path_plot.opts(opts.Path(active_tools=['freehand_draw']))
        self.pointer_pos.source = self.path_plot
        self.clicked_pos.source = self.path_plot
        self.zoom_range.source = self.path_plot
        self.plot_size.source = self.path_plot

        self.path_plot = self.path_plot * hv.DynamicMap(self.plot_pointer)

    @param.depends('zoom_range.x_range', 'plot_size.width', watch=True)
    def monitor_zoom_level(self):
        # TODO handle other than xy axis

        plot_width = self.plot_size.width
        if plot_width:
            zrange = self.zoom_range.x_range

            if np.isnan(zrange[0]) or np.isnan(zrange[1]):
                return

            elif zrange is None:
                self.zoom_level = plot_width / self.dataset.img.shape[1]

            else:
                zoomed_width = self.dataset.spacing[1] * (zrange[1] -
                                                          zrange[0])
                self.zoom_level = plot_width / zoomed_width

    @param.depends('dataset.drawing_label', 'tool_width', 'zoom_level')
    def plot_path(self, data):
        self.a = self.dataset.drawing_label
        # at this stage is redered and size known
        if not self.zoom_initialized:
            self.monitor_zoom_level()
            self.zoom_initialized = True

        path = hv.Path(data)
        path.opts(
            opts.Path(line_width=self.tool_width * self.zoom_level + 1,
                      color=self.cmap[self.dataset.drawing_label],
                      line_cap='round',
                      line_join='round'))

        return path

    @param.depends('dataset.drawing_label', 'tool_width', 'zoom_level',
                   'pointer_pos.x', 'pointer_pos.y')
    def plot_pointer(self):

        if not self.zoom_initialized:
            self.monitor_zoom_level()
            self.zoom_initialized = True

        pos_x = self.pointer_pos.x
        if pos_x is None:
            pos_x = 0.

        pos_y = self.pointer_pos.y
        if pos_y is None:
            pos_y = 0.

        pt = hv.Points((pos_x, pos_y))
        pt.opts(
            opts.Points(
                size=self.tool_width * self.zoom_level,
                color=self.cmap[self.dataset.drawing_label],
                shared_axes=True,
            ))

        return pt

    @staticmethod
    def _get_axis_id(axis_name):
        return {'z': 0, 'y': 1, 'x': 2}[axis_name]

    @param.depends('freehand.data', watch=True)
    def embedd_path(self):
        '''write the polygon path on rasterized array with correct label and width'''

        coords = self.freehand.data
        if coords and coords['ys']:
            xs = np.asarray(np.rint(np.asarray(coords['xs'][0]) - 0.5),
                            dtype=int)
            ys = np.asarray(np.rint(np.asarray(coords['ys'][0]) - 0.0),
                            dtype=int)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)

            mask = np.zeros_like(self.dataset.img, np.uint8)
            if mask.ndim > 2:
                axis = self._get_axis_id(self.slicer.axis)
                loc = [slice(None) for _ in range(mask.ndim)]
                loc[axis] = self.slicer.slice_id

                cv.polylines(
                    mask[loc],
                    [pts],
                    False,
                    1,
                    self.tool_width,  # // 2,
                    cv.LINE_8)
            else:
                # draw polyline on minimal size crop
                margin = self.tool_width // 2 + 1
                loc = [
                    slice(max(0, pts[:, ax].min() - margin),
                          pts[:, ax].max() + margin) for ax in range(2)
                ]
                offset = np.array([s.start for s in loc])[None]
                loc = loc[::-1]
                submask = mask[loc]
                cv.polylines(
                    submask,
                    [pts - offset],
                    False,
                    1,
                    self.tool_width,  # // 2,
                    cv.LINE_8)

            mask = mask.astype(bool)

            self.dataset.write_label(mask)
            self._clear()

    @param.depends('clicked_pos.x', 'clicked_pos.y', watch=True)
    def monitor_clicked_label(self):
        if self.clicked_pos.x is not None and self.clicked_pos.y is not None:
            x = int(round(self.clicked_pos.x))
            y = int(round(self.clicked_pos.y))
            coords = (y, x)

            if self.dataset.img.ndim > 2:
                axis = self._get_axis_id(self.slicer.axis)
                coords = np.insert(np.array(coords), axis,
                                   self.slicer.slice_id)

            self.dataset.click_callback(coords)

    def _clear(self):
        self.pipe.send([])

    def widgets(self):
        wg = self.dataset.widgets()
        wg.append(self.param.tool_width)
        return wg
