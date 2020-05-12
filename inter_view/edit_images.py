import numpy as np
import cv2 as cv
import param
import panel as pn
import holoviews as hv

from holoviews import streams, opts

from inter_view.utils import label_cmap

# TODO
# handle spacing
# tool size max bound as function image size
# handle multiple axis
#     spacing
#     xy, z drawing size
# TODO client side optimization, javascript links instead of python
#   label dropdown to pointer+path color
#   tool size to pointer+path size (can access poltting range to adjsut for zoom level?)
#   position of tool glyph preview to mouse position


class LabelEditor(param.Parameterized):
    '''Extract a data array from a holoviews element and makes it editable'''

    array = param.Array(precedence=-1)
    kdims_val = param.List(precedence=-1)
    locked_mask = param.Array(
        precedence=-1, doc='''mask of region that should not be updated''')

    drawing_label = param.Selector(default=1, objects=[-1, 0, 1])
    img_pipe = param.Parameter(streams.Pipe(), precedence=-1)
    editor_switches = param.ListSelector(
        default=[], objects=['label picker', 'lock bg', 'lock fg'])

    update_inprogress = param.Boolean(False, precedence=-1)

    def __call__(self, dmap):

        dmap = hv.util.Dynamic(dmap,
                               operation=self._dynamic_call,
                               streams=[self.img_pipe]).relabel(dmap.label)
        return dmap

    def _dynamic_call(self, element, data):
        '''rebuilds image with piped data'''

        if data is not None and self.update_inprogress:
            # element is being updated with newly drawn path

            element = element.clone((*self.kdims_val, data))

        else:
            # element is being update from outside
            # reset internal data with new element and let pass through
            # TODO handle multiple vdims?
            kdims_name = [d.name for d in element.kdims]
            self.kdims_val = [
                element.dimension_values(name, expanded=False)
                for name in kdims_name
            ]
            self.array = element.dimension_values(element.vdims[0].name,
                                                  flat=False)

        return element

    @param.depends('array', 'editor_switches', watch=True)
    def update_locked_mask(self):
        mask = np.zeros_like(self.array, dtype=bool)

        if 'lock bg' in self.editor_switches:
            mask[self.array == 0] = True

        if 'lock fg' in self.editor_switches:
            mask[self.array > 0] = True

        self.locked_mask = mask

    def write_label(self, mask):

        new_array = self.array
        new_array[mask & (~self.locked_mask)] = self.drawing_label

        # assign new array to trigger updates
        self.array = new_array

        self.img_pipe.send(self.array)

    @param.depends('array', watch=True)
    def update_drawing_label_list(self):
        '''List of label to choose from.'''

        unique_labels = np.unique(self.array)
        # add an extra label to annotate new objects
        unique_labels = np.append(unique_labels, unique_labels.max() + 1)
        unique_labels = list({-1, 0, 1}.union(set(unique_labels)))
        unique_labels.sort()

        # if current label not in new list, set to -1
        if self.drawing_label not in unique_labels:
            self.drawing_label = -1

        self.param.drawing_label.objects = unique_labels

    def set_picked_label(self, label):
        if 'label picker' in self.editor_switches:
            self.drawing_label = label

    def delete_label(self, event=None):
        self.array[self.array == self.drawing_label] = -1
        self.array = self.array
        self.img_pipe.send(self.array)

    def widgets(self):
        delete_button = pn.widgets.Button(name='delete selected label')
        delete_button.on_click(self.delete_label)

        return pn.WidgetBox(
            pn.Param(self.param,
                     widgets={
                         'editor_switches': {
                             'type': pn.widgets.CheckButtonGroup
                         }
                     }), delete_button)


class FreehandEditor(param.Parameterized):
    '''Adds a freehand drawing tool that embeds the drawn path in the iamge/stack'''

    label_editor = param.Parameter(precedence=-1)
    freehand = param.Parameter(streams.FreehandDraw(num_objects=1),
                               precedence=-1)
    pointer_pos = param.Parameter(streams.PointerXY(), precedence=-1)
    clicked_pos = param.Parameter(streams.SingleTap(transient=True),
                                  precedence=-1)

    zoom_range = param.Parameter(
        streams.RangeX(),
        doc=
        '''range stream used to adjust glyph size based on zoom level, assumes data_aspect=1''',
        precedence=-1)
    zoom_level = param.Number(1.0, precedence=-1)
    zoomed_initialized = param.Boolean(False, precedence=-1)

    drawingtool = param.Parameter(precedence=-1)
    slicing_info = param.Callable(
        doc='''callable return the axis and slice id of the current plot''',
        precedence=-1)
    cmap = param.Parameter(label_cmap, precedence=-1)

    tool_width = param.Integer(20, bounds=(1, 300))
    _plot_width = param.Integer(500, precedence=-1)

    def __init__(self, label_editor, plot_width, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_editor = label_editor
        self._plot_width = plot_width

        # grey glyph for drawing label -1 (unlabeled)
        self.cmap[-1] = '#999999'

        self.reset()

    def reset(self):
        '''reset drawtool plots'''

        self.label_editor.kdims_val = []
        self.zoomed_initialized = False

        self.pipe = streams.Pipe(data=[])
        self.drawingtool = hv.DynamicMap(self.plot_path, streams=[self.pipe])
        self.freehand.source = self.drawingtool

        self.drawingtool.opts(opts.Path(active_tools=['freehand_draw']))

        self.pointer_pos.source = self.drawingtool
        self.clicked_pos.source = self.drawingtool

        self.drawingtool = self.drawingtool * hv.DynamicMap(self.plot_pointer)

    @param.depends('zoom_range.x_range', watch=True)
    def monitor_zoom_level(self):
        zrange = self.zoom_range.x_range

        if self.label_editor.kdims_val:
            image_px_width = len(self.label_editor.kdims_val[0])
        else:
            image_px_width = self._plot_width

        if zrange is None:
            self.zoom_level = self._plot_width / image_px_width

        else:
            zoom_width = zrange[1] - zrange[0]
            # TODO handle other than xy axis
            full_width = self.label_editor.kdims_val[0].max(
            ) - self.label_editor.kdims_val[0].min()
            self.zoom_level = full_width / zoom_width * self._plot_width / image_px_width

    @param.depends('label_editor.drawing_label', 'tool_width', 'zoom_level')
    def plot_path(self, data):
        # at this stage is redered and size known
        if not self.zoomed_initialized:
            self.monitor_zoom_level()
            self.zoomed_initialized = True

        path = hv.Path(data)
        path.opts(
            opts.Path(line_width=self.tool_width * self.zoom_level + 1,
                      color=self.cmap[self.label_editor.drawing_label],
                      line_cap='round',
                      line_join='round'))

        return path

    @param.depends('label_editor.drawing_label', 'tool_width', 'zoom_level',
                   'pointer_pos.x', 'pointer_pos.y')
    def plot_pointer(self):

        if not self.zoomed_initialized:
            self.monitor_zoom_level()
            self.zoomed_initialized = True

        # limit to image bounds
        if self.label_editor.kdims_val:
            x_max = self.label_editor.kdims_val[0].max()
            y_max = self.label_editor.kdims_val[1].max()
        else:
            x_max = 0.
            y_max = 0.

        pos_x = self.pointer_pos.x
        if pos_x is None:
            pos_x = 0.

        pos_y = self.pointer_pos.y
        if pos_y is None:
            pos_y = 0.

        pos_x = max(min(pos_x, x_max), 0)
        pos_y = max(min(pos_y, y_max), 0)

        pt = hv.Points((pos_x, pos_y))
        pt.opts(
            opts.Points(
                size=self.tool_width * self.zoom_level,
                color=self.cmap[self.label_editor.drawing_label],
                shared_axes=True,
            ))

        return pt

    @param.depends('clicked_pos.x', 'clicked_pos.y', watch=True)
    def monitor_clicked_label(self):
        if self.clicked_pos.x is not None and self.clicked_pos.y is not None:
            # TODO update to handle spacing/3D

            axis, slice_id = self.slicing_info()
            x = int(round(self.clicked_pos.x))
            y = int(round(self.clicked_pos.y))

            if self.label_editor.array.ndim > 2:
                self.label_editor.set_picked_label(
                    int(self.label_editor.array[slice_id, y, x]))
            else:
                self.label_editor.set_picked_label(
                    int(self.label_editor.array[y, x]))

    def __call__(self, dmap, slicing_info):
        self.slicing_info = slicing_info
        self.zoom_range.source = dmap

        return dmap

    def embedd_path(self):
        '''write the polygon path on rasterized array with correct label and width'''

        coords = self.freehand.data
        xs = np.asarray(np.rint(np.asarray(coords['xs'][0]) - 0.5), dtype=int)
        ys = np.asarray(np.rint(np.asarray(coords['ys'][0]) - 0.0), dtype=int)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)

        mask = np.zeros_like(self.label_editor.array, np.uint8)
        if mask.ndim > 2:
            axis, slice_id = self.slicing_info()
            cv.polylines(
                mask[slice_id],
                [pts],
                False,
                1,
                self.tool_width,  # // 2,
                cv.LINE_8)
        else:
            cv.polylines(
                mask,
                [pts],
                False,
                1,
                self.tool_width,  # // 2,
                cv.LINE_8)

        mask = mask.astype(bool)

        self.label_editor.write_label(mask)

    @param.depends('freehand.data', watch=True)
    def _update_data(self):
        coords = self.freehand.data

        if coords and coords['ys'] and not self.label_editor.update_inprogress:
            self.label_editor.update_inprogress = True

            self.embedd_path()

            self._clear()
            self.label_editor.update_inprogress = False

    def _clear(self):
        self.pipe.send([])

    def widgets(self):
        return pn.WidgetBox(pn.Param(self.param))
