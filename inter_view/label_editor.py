import numpy as np
import os
from glob import glob

from inter_view.utils import min_max_scaling

from bokeh.models import ColumnDataSource
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Select
from bokeh.layouts import widgetbox, row
from bokeh.models.widgets import RadioButtonGroup, Button, Toggle, CheckboxGroup
from bokeh.models import Slider
from bokeh.events import Tap

from skimage.io import imread, imsave
from skimage.segmentation import relabel_sequential

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label as ndimage_label
from scipy.ndimage.morphology import distance_transform_edt


class DataHandler:
    '''maintains images list and loads and save them from/on disk'''

    def __init__(self, base_dir, master_channel, config, n_images=10, seed=13):

        self.n_images = n_images
        self.base_dir = base_dir
        self.master = master_channel
        self.channels = list(config.keys())

        self.paths = {}
        self.preproc = {}
        self.images = {}  # loaded images
        self.sequential = {
        }  # whether image contains labels that should remain sequential

        for ch, val in config.items():
            self.preproc[ch] = val.get('preproc', lambda x: x)
            self.sequential[ch] = val.get('sequential', False)

            self.paths[ch] = os.path.join(base_dir, val['path'])
            if not os.path.exists(self.paths[ch]):
                os.makedirs(self.paths[ch])

        self.filenames = glob(os.path.join(self.paths[self.master], '*.tif'))
        self.filenames = sorted([os.path.basename(p) for p in self.filenames])
        if n_images:
            np.random.seed(seed)
            np.random.shuffle(self.filenames)
            self.filenames = self.filenames[0:n_images]
            self.filenames.sort()

        # ~ print(self.filenames)

    def load_images(self, idx):
        self.idx = idx

        self.images[self.master] = imread(
            os.path.join(self.paths[self.master], self.filenames[idx]))
        self.images[self.master] = self.preproc[self.master](
            self.images[self.master])

        for c in self.channels:
            if c != self.master:
                try:
                    self.images[c] = imread(
                        os.path.join(self.paths[c], self.filenames[idx]))
                except:
                    # is image doesn't exist yet, initialize with same size as master with label=-1
                    self.images[c] = np.zeros_like(self.images[self.master],
                                                   dtype=np.int16) - 1
                self.images[c] = self.preproc[c](self.images[c])

            if self.sequential[c]:
                self.relabel_sequential(c)

    def save_channel(self, ch):
        imsave(os.path.join(self.paths[ch], self.filenames[self.idx]),
               self.images[ch])

    def delete_val_from_channel(self, val, ch):
        self.images[ch][self.images[ch] == val] = -1
        if val > 0:  # label 0 reserved for background, even if not sequential, no need to relabel
            self.relabel_sequential(ch)

    def relabel_sequential(self, ch):
        '''relable sequential, reserving -1,0 labels if not existing (separate current code in fct)'''

        bg_unlabeled = False
        if 0 not in self.images[ch]:
            bg_unlabeled = True

        self.images[ch][:], _, _ = relabel_sequential(self.images[ch] + 1)
        self.images[ch] -= 1
        if bg_unlabeled:  # 0 should remain unused
            self.images[ch][self.images[ch] >= 0] += 1

    def map_label(self, ch, ch_target):
        '''maps ch labels to ch_target labels with largest overlap, assign 
        new label when only overlapping with undefined areas'''

        img = self.images[ch]
        img_target = self.images[ch_target]

        if img.max() >= 0:
            lut = [-1] * (img.max() + 1)
            extra_label_start = img_target.max() + 1
            extra_label_count = 0

            for img_label in np.unique(img):
                img_target_labels, intersections = np.unique(
                    img_target[img == img_label], return_counts=True)
                if img_target_labels[0] == -1:
                    img_target_labels = np.delete(img_target_labels, 0)
                    intersections = np.delete(intersections, 0)

                if img_target_labels.size > 0:
                    lut[img_label] = img_target_labels[np.argmax(
                        intersections)]
                else:
                    lut[img_label] = extra_label_start + extra_label_count
                    extra_label_count += 1

            print(lut)

            # apply lut
            lut = np.asarray(lut)
            img[:] = lut[img]

    def merge_channels(self, ch_primary, ch_secondary):
        '''replace secondary channel by a composite: primary_ch where defined (i.e. >= 0), secondary elsewhere.'''

        defined_idxs = self.images[ch_primary] >= 0
        self.images[ch_secondary][defined_idxs] = self.images[ch_primary][
            defined_idxs]


# TODO
# calc distance transform only in neighborhood of seed (defined by threhsold)
# checkbox to auto increment label after 'apply'
#    - set drawing color
#   - add callback in global control (watch for inf loop): self.view.top.draw_r.glyph.on_change('line_color', cb)
class MaskControls:
    '''Widget layout and callbacks to show thresholded mask output and apply it on annotations'''

    def __init__(self, datahandler, view, mask_channel, source_channel,
                 drawing_channel):
        self.datahandler = datahandler
        self.view = view
        self.ch = mask_channel
        self.source_ch = source_channel
        self.drawing_ch = drawing_channel
        self.update_in_progress = False

        self.button_toggle_seed = Toggle(label='seeded segmentation OFF',
                                         active=False)
        self.button_toggle_seed.on_click(self.toggle_seed)

        self.view.add_tap_callback(self.callback_set_seed_location)
        self.seed_pos = (0, 0, 0)

        self.button_apply = Button(label="apply", button_type='primary')
        self.button_apply.on_click(self.apply)

        # callback policy on apply to JS --> hacky workaround with fake datasource:
        # https://stackoverflow.com/questions/38375961/throttling-in-bokeh-application/38379136#38379136
        self.threshold_source = ColumnDataSource(data=dict(value=[1]))
        self.threshold_source.on_change('data', self.callback_slider_threshold)
        self.slider_theshold = Slider(title='threshold',
                                      start=0,
                                      end=1,
                                      value=0.5,
                                      step=0.001,
                                      callback_policy='mouseup')
        self.slider_theshold.callback = CustomJS(
            args=dict(threshold_source=self.threshold_source),
            code="""
                threshold_source.data = { value: [cb_obj.value] }""")

        # callback policy on apply to JS --> hacky workaround with fake datasource:
        # https://stackoverflow.com/questions/38375961/throttling-in-bokeh-application/38379136#38379136
        self.blur_source = ColumnDataSource(data=dict(value=[1]))
        self.blur_source.on_change('data', self.callback_slider_blur)
        self.slider_blur = Slider(title='blur',
                                  start=0,
                                  end=10,
                                  value=0.2,
                                  step=0.05,
                                  callback_policy='mouseup')
        self.slider_blur.callback = CustomJS(
            args=dict(blur_source=self.blur_source),
            code="""
                blur_source.data = { value: [cb_obj.value] }""")

        # callback policy on apply to JS --> hacky workaround with fake datasource:
        # https://stackoverflow.com/questions/38375961/throttling-in-bokeh-application/38379136#38379136
        self.dist_reg_source = ColumnDataSource(data=dict(value=[1]))
        self.dist_reg_source.on_change('data', self.callback_slider_dist_reg)
        self.slider_dist_reg = Slider(title='distance regularizer',
                                      start=0,
                                      end=1,
                                      value=0.,
                                      step=0.01,
                                      callback_policy='mouseup')
        self.slider_dist_reg.callback = CustomJS(
            args=dict(dist_reg_source=self.dist_reg_source),
            code="""
                dist_reg_source.data = { value: [cb_obj.value] }""")

        self.checkbox_invert = CheckboxGroup(labels=['invert'], active=[])
        self.checkbox_invert.on_click(self.callback_checkbox_invert)

        self.checkbox_apply_params = CheckboxGroup(
            labels=['whole stack', 'override'], active=[0])

        self.l = widgetbox([
            self.button_toggle_seed,
            row(self.checkbox_apply_params, self.button_apply, width=300),
            self.slider_theshold,
            self.checkbox_invert,
            self.slider_blur,
            self.slider_dist_reg,
        ],
                           width=300)

        # init cached blurred image and update when source image changes
        self.callback_update_image(None, None, None)
        self.view.add_image_on_change_callback(self.source_ch,
                                               self.callback_update_image)

        # init distance to seed
        self.dist_to_seed = np.zeros_like(self.blurred_img)

    def callback_update_image(self, attr, old, new):
        if not self.view.update_in_progress:
            self.callback_slider_blur(None, None, self.blur_source.data)

    def toggle_seed(self, state):
        if state:
            self.button_toggle_seed.label = 'seeded segmentation ON'
        else:
            self.button_toggle_seed.label = 'seeded segmentation OFF'

        self.update_segmentation()

    def apply(self):
        '''Write current mask on drawing channel'''

        mask = self.mask.copy()
        if 1 not in self.checkbox_apply_params.active:
            #  don't ovverride existing annotations
            mask[self.datahandler.images[self.drawing_ch] >= 0] = False

        if 0 not in self.checkbox_apply_params.active:
            # single slice
            sl = self.view.get_zslice()
            self.datahandler.images[self.drawing_ch][sl][
                mask[sl]] = self.view.get_drawing_color()
        else:
            self.datahandler.images[
                self.drawing_ch][mask] = self.view.get_drawing_color()

        self.view.force_update(self.drawing_ch)

    def callback_slider_threshold(self, attr, old, new):
        # ~ thresh = new['value'][0]
        self.update_segmentation()

    def callback_slider_dist_reg(self, attr, old, new):
        # ~ thresh = new['value'][0]
        self.update_segmentation()

    def callback_slider_blur(self, attr, old, new):
        sigma = self.slider_blur.value
        sampling = self.view.get_sampling()

        if sigma > 0:
            self.blurred_img = gaussian_filter(
                self.datahandler.images[self.source_ch],
                sigma=sigma / sampling)
        else:
            self.blurred_img = self.datahandler.images[self.source_ch]

        self.blurred_img = self.blurred_img.astype(np.float32)
        self.blurred_img /= self.blurred_img.max()
        self.update_segmentation()

    def callback_checkbox_invert(self, active):
        self.update_segmentation()

    def callback_set_seed_location(self, event):
        if self.button_toggle_seed.active:
            z = self.view.get_zslice()
            x = int(round(event.y))
            y = int(round(event.x))
            self.seed_pos = (z, x, y)

            sampling = self.view.get_sampling()
            sampling = sampling / sampling.min()
            seed_mask = np.ones_like(self.blurred_img, dtype=bool)
            seed_mask[self.seed_pos] = False
            self.dist_to_seed = distance_transform_edt(seed_mask,
                                                       sampling=sampling)

            self.dist_to_seed = np.clip(self.dist_to_seed, 0.,
                                        float(max(seed_mask.shape)))
            self.dist_to_seed /= float(max(seed_mask.shape))

            self.update_segmentation()

    def callback_update_segmentation(self, attr, old, new):
        self.update_segmentation()

    def update_segmentation(self):
        '''Update seeded segmentation image'''

        if self.update_in_progress == False:
            self.update_in_progress = True

            if self.checkbox_invert.active:  # dark object on white background
                composite = 1. - self.blurred_img
            else:
                composite = self.blurred_img

            if self.button_toggle_seed.active and self.blurred_img.shape == self.dist_to_seed.shape:
                composite = (1 - self.slider_dist_reg.value
                             ) * composite + self.slider_dist_reg.value * (
                                 1. - self.dist_to_seed)

            self.mask = composite >= self.slider_theshold.value

            if self.button_toggle_seed.active:
                labels, _ = ndimage_label(self.mask)
                seed_label = labels[self.seed_pos]
                if seed_label > 0:
                    self.mask = labels == seed_label
                else:
                    self.mask[:] = False

            self.datahandler.images[self.ch][:] = self.mask
            self.view.force_update(self.ch)
            self.update_in_progress = False


class Controls:
    '''Widget layout and callbacks to draw on images'''

    def __init__(self, datahandler, view, drawing_channel):
        self.datahandler = datahandler
        self.view = view
        self.drawing_ch = drawing_channel
        self.view.add_drawing_tools(drawing_channel, self.callback_on_drawing)

        self.save_button = Button(label="Save annotations",
                                  button_type='primary')
        self.save_button.on_click(self.callback_save_button)

        self.discard_button = Button(label="Discard changes",
                                     button_type='primary')
        self.discard_button.on_click(self.callback_discard_button)

        self.file_select = Select(
            title="Save current and load new file",
            value=str(datahandler.idx),
            options=[(idx, fn)
                     for idx, fn in enumerate(datahandler.filenames)])
        self.file_select.on_change('value', self.callback_file_select)

        # toggle switch to pick label on click
        self.button_toggle_label_picker = Toggle(label='Label picker ON',
                                                 active=True)
        self.button_toggle_label_picker.on_click(self.toggle_picker)
        self.view.add_tap_callback(self.callback_pick_label)

        self.del_label_button = Button(label="Delete selected label",
                                       button_type='primary')
        self.del_label_button.on_click(self.callback_del_label)

        self.channel_select_button = RadioButtonGroup(
            labels=self.datahandler.channels, active=2)
        self.channel_select_button.on_click(self.callback_channel_select)

        self.label_select = Select(title="Selected label",
                                   value='-1',
                                   options=['-1'])
        self.label_select.on_change('value', self.callback_label_select)
        self.update_label_select_list()

        self.tool_size_slider = Slider(title='tool size',
                                       start=1,
                                       end=100,
                                       value=self.view.get_toolsize(),
                                       step=1)
        self.tool_size_slider.on_change('value', self.callback_tool_size)

        self.alpha_slider = Slider(title='channel alpha',
                                   start=0,
                                   end=1.,
                                   value=0.5,
                                   step=0.01)
        self.alpha_slider.on_change('value', self.callback_ch_alpha)

        self.l = widgetbox([
            row(self.button_toggle_label_picker, width=600),
            self.channel_select_button,
            self.tool_size_slider,
            self.alpha_slider,
            self.label_select,
            self.del_label_button,
            self.file_select,
            self.save_button,
            self.discard_button,
        ],
                           width=600)

    def callback_save_button(self, button):
        self.datahandler.save_channel(self.drawing_ch)

    def callback_del_label(self, button):

        # ~ self.datahandler.delete_val_from_channel(int(self.label_select.value),
        # ~ 'annotations')
        # ~ self.datahandler.delete_val_from_channel(int(self.label_select.value),
        # ~ 'predictions')
        # ~ self.update_label_select_list()
        # ~ self.orthoview.top.force_update('annotations')
        # ~ self.orthoview.top.force_update('predictions')

        self.datahandler.delete_val_from_channel(int(self.label_select.value),
                                                 self.drawing_ch)
        self.update_label_select_list()
        self.view.force_update(self.drawing_ch)

    def callback_discard_button(self, button):
        #reload current img
        self.load_and_display(self.datahandler.idx)

    def callback_file_select(self, attr, old, new):
        print('loading img', new)

        self.datahandler.save_channel(self.drawing_ch)
        self.load_and_display(int(new))

    def load_and_display(self, idx):
        self.datahandler.load_images(idx)
        self.view.set_images(self.datahandler.images)
        self.update_label_select_list()

    def toggle_picker(self, state):
        if state:
            self.button_toggle_label_picker.label = 'Label picker ON'
        else:
            self.button_toggle_label_picker.label = 'Label picker OFF'

    def callback_pick_label(self, event):
        if self.button_toggle_label_picker.active:
            z = self.view.get_zslice()
            x = int(round(event.y))
            y = int(round(event.x))
            ch = self.channel_select_button.labels[
                self.channel_select_button.active]
            self.label_select.value = str(
                self.view.get_npimage(ch)[:, ::-1][z, x, y])

    def callback_label_select(self, attr, old, new):
        '''updates drawing color attribute view renderer'''
        self.view.set_drawing_color(int(new))

    def callback_on_drawing(self, attr, old, new):
        ''''''
        self.update_label_select_list()

    def update_label_select_list(self):
        '''List of label to choose from.'''

        ch = self.channel_select_button.labels[
            self.channel_select_button.active]
        unique_labels = np.unique(self.datahandler.images[ch])
        # add an extra label to annotate new objects
        unique_labels = np.append(unique_labels, unique_labels.max() + 1)
        unique_labels = list({-1, 0}.union(set(unique_labels)))
        unique_labels.sort()

        self.label_select.options = [str(l) for l in unique_labels]
        if self.label_select.value not in self.label_select.options:
            self.label_select.value = '-1'

    def callback_tool_size(self, attr, old, new):
        self.view.set_toolsize(new)

    def callback_ch_alpha(self, attr, old, new):
        '''update alpha of currently selected channel'''

        ch = self.channel_select_button.labels[
            self.channel_select_button.active]
        self.view.set_channel_alpha(ch, new)

    def callback_channel_select(self, active):
        '''updates selected channel'''
        channel = self.channel_select_button.labels[active]
        self.update_label_select_list()

        self.alpha_slider.value = self.view.get_channel_alpha(channel)
