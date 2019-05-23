import numpy as np
import pandas as pd


from bokeh.plotting import figure
from bokeh.models import Range1d, ColorBar, WheelZoomTool
from bokeh.models import BoxSelectTool, LassoSelectTool
from bokeh.models import ColumnDataSource, Legend, LegendItem
from bokeh.layouts import gridplot, layout, row
from bokeh.transform import linear_cmap
from bokeh.palettes import viridis, d3, brewer, inferno

from inter_view.utils import update_color_mapping

# TODO
# consistent naming: plot(p), renderer(r), glyph(g), layout(l), etc.
# outliers selection (when hue=outliers) + save_df
# marginal plots: if hue is catgorical, plot kde with diferent colors
# barplots of current hue count (category) or histogram (numeric)
# data table
# make image tabs into a bokeh composition --> propagate update functions
# menu for scatter size selection --> autoscale to reasonable values
# expose plot config like: scatter alpha, linear/log axis
# separate legend and colorbar


class JointPlot():
    '''Creates a scatter plot with marginal histograms.
    '''

    def __init__(self,
                 source,
                 x_vals,
                 y_vals,
                 hue='disabled',
                 tooltips_columns=None,
                 **kwargs):

        self.source = source
        self.x_vals = x_vals
        self.y_vals = y_vals

        self.hue = hue
        param = update_color_mapping(source, hue)
        self.hue_type = param['hue_type']
        self.color_map = param['cmap']

        self.tooltips_formatting = [
            ("(x,y)", "($x{0.}, $y{0.})"),
        ]
        if tooltips_columns:
            self.tooltips_formatting += [(s.replace('_', ' '), '@' + s)
                                         for s in tooltips_columns]

        self.scatter_config = {
            'fill_color': 'color',
            'fill_alpha': 1.,
            'line_width': 0,
            'line_alpha': 0,
            'size': 5,
            'nonselection_fill_color': 'color',
            'nonselection_fill_alpha': 0.2,
            'nonselection_line_alpha': 0.0,
            'selection_fill_color': 'color',
            'selection_fill_alpha': 1.0,
            'selection_line_alpha': 1.0,
            'selection_line_color': 'white',
        }

        self.plot()

    def update_hue(self, hue_type, cmap):

        self.hue_type = hue_type

        if self.hue_type == 'numeric':
            self.ps.select(ColorBar).color_mapper = cmap['transform']
            self.ps.legend[0].items = []

        elif self.hue_type == 'category':
            self.ps.select(ColorBar).color_mapper = None
            if self.ps.legend[0].items:  # legend item already exist
                self.ps.legend[0].items[0] = LegendItem(
                    label={'field': cmap['field']}, renderers=[self.scatter])
            else:
                self.ps.legend[0].items.append(
                    LegendItem(label={'field': cmap['field']},
                               renderers=[self.scatter]))

        else:
            self.ps.select(ColorBar).color_mapper = None
            self.ps.legend[0].items = []

        # recompute histo for same variable but different grouping
        self.update_x(self.x_vals)
        self.update_y(self.y_vals)

    def update_x(self, x_vals):
        self.x_vals = x_vals

        hist = self.compute_histogram(self.x_vals)
        self.histo_h.data_source.data = hist.data

        self.scatter.glyph.x = x_vals
        self.ph.y_range.end = max(hist.data['count']) * 1.1
        self.ph.xaxis.axis_label = self.x_vals.replace('_', ' ')

    def update_y(self, y_vals):
        self.y_vals = y_vals

        hist = self.compute_histogram(self.y_vals)
        self.histo_v.data_source.data = hist.data

        self.scatter.glyph.y = y_vals
        self.pv.x_range.end = max(hist.data['count']) * 1.1
        self.pv.yaxis.axis_label = self.y_vals.replace('_', ' ')

    def plot(self):

        self.plot_scatter()
        self.plot_vertical_histo()
        self.plot_horizontal_histo()
        self.p = gridplot([[self.ps, self.pv], [self.ph, None]],
                          merge_tools=False)

    def plot_scatter(self):
        self.ps = figure(
            tools='tap,pan,wheel_zoom, box_zoom,box_select,lasso_select,reset',
            tooltips=self.tooltips_formatting,
            active_drag='box_select',
            active_scroll='wheel_zoom',
            toolbar_location='above',
            plot_width=600,
            plot_height=600,
            # ~ plot_width=p.plot_width+40, # why 40 difference to line up plot in layout????
            x_axis_location=None,
            y_axis_location=None)

        self.ps.select(BoxSelectTool).select_every_mousemove = True
        self.ps.select(LassoSelectTool).select_every_mousemove = True

        self.ps.border_fill_color = None
        self.ps.outline_line_color = None

        self.scatter = self.ps.scatter(x=self.x_vals,
                                       y=self.y_vals,
                                       source=self.source,
                                       legend=self.hue,
                                       **self.scatter_config)

        # necessary render a colorbar once to display properly (can be disable later by settign color_mapper to None)
        place_holder_color_map = linear_cmap(field_name=None,
                                             palette=viridis(256),
                                             low=0,
                                             high=0)
        color_bar = ColorBar(color_mapper=place_holder_color_map['transform'],
                             width=8,
                             label_standoff=5,
                             border_line_color=None,
                             location=(0, 0),
                             major_label_text_font_size='6pt',
                             major_label_text_align='left')
        self.ps.add_layout(color_bar, 'left')

        if self.hue_type == 'numeric':
            color_bar.color_mapper = self.color_map['transform']
            self.ps.legend[0].items = []
        else:
            self.resize_legend(self.ps.legend)

    def plot_vertical_histo(self):

        histo_source = self.compute_histogram(self.y_vals)

        self.pv = figure(toolbar_location=None,
                         plot_width=self.ps.plot_width // 3,
                         plot_height=self.ps.plot_height,
                         x_range=(0, max(histo_source.data['count']) * 1.1),
                         y_range=self.ps.y_range,
                         min_border=0,
                         y_axis_location='right',
                         x_axis_location='above')

        self.pv.border_fill_color = None
        self.pv.outline_line_color = None
        self.pv.axis.axis_label_text_font_size = '15pt'
        self.pv.axis.major_label_text_font_size = '7pt'
        self.pv.xaxis.major_label_orientation = -np.pi / 4
        self.pv.yaxis.axis_label = self.y_vals.replace('_', ' ')
        self.histo_v = self.pv.quad(left=0.1,
                                    bottom='left',
                                    top='right',
                                    right='count',
                                    source=histo_source,
                                    fill_color='color',
                                    fill_alpha=0.5,
                                    line_color=None,
                                    line_width=0)

    def plot_horizontal_histo(self):

        histo_source = self.compute_histogram(self.x_vals)

        self.ph = figure(toolbar_location=None,
                         plot_width=self.ps.plot_height,
                         plot_height=self.ps.plot_width // 3,
                         x_range=self.ps.x_range,
                         y_range=(0, max(histo_source.data['count']) * 1.1),
                         min_border=0,
                         y_axis_location="left")

        self.ph.border_fill_color = None
        self.ph.outline_line_color = None
        self.ph.axis.axis_label_text_font_size = '15pt'
        self.ph.axis.major_label_text_font_size = '7pt'
        self.ph.yaxis.major_label_orientation = -np.pi / 4
        self.ph.xaxis.axis_label = self.x_vals.replace('_', ' ')
        self.histo_h = self.ph.quad(bottom=0.1,
                                    left='left',
                                    right='right',
                                    top='count',
                                    source=histo_source,
                                    fill_color='color',
                                    fill_alpha=0.5,
                                    line_color=None,
                                    line_width=0)

    def compute_histogram(self, name):
        '''Returns a datasource containing histogram and edges of column name passed in argument
        '''

        if self.hue_type == 'category':
            df = pd.DataFrame(
                {k: self.source.data[k]
                 for k in (name, 'color')})
            bins = pd.cut(df[name], 50)
            histo = df.groupby([bins, 'color']).count()
            histo.index.rename(['bins', 'color'], inplace=True)
        else:
            df = pd.Series(self.source.data[name])
            histo = df.value_counts(bins=50)
            histo.index.rename('bins', inplace=True)
            histo = histo.to_frame()

        histo.columns = ['count']
        histo = histo.reset_index().dropna()
        histo['left'] = [inter.left for inter in histo['bins']]
        histo['right'] = [inter.right for inter in histo['bins']]
        histo.drop(columns='bins', inplace=True)

        if self.hue_type != 'category':
            histo['color'] = 'grey'

        return ColumnDataSource(data=histo)

    @staticmethod
    def resize_legend(legend_handle):
        legend_handle.click_policy = 'hide'
        legend_handle.glyph_width = 15
        legend_handle.glyph_height = 15
        legend_handle.label_standoff = 5
        legend_handle.spacing = 0
        legend_handle.padding = 0
        legend_handle.label_text_font_size = '8pt'
        legend_handle.label_text_baseline = 'middle'
        legend_handle.label_height = 15
        legend_handle.label_width: 25

    # TODO kde plot
    # from sklearn.neighbors import KernelDensity
    # from sklearn.model_selection import GridSearchCV

    # ~ self.kde_sklearn(self.source.data[name])

    # @staticmethod
    # def kde_sklearn(x):
    # # https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

    # def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    # kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    # kde_skl.fit(x[:, np.newaxis])
    # # score_samples() returns the log-likelihood of the samples
    # log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    # return np.exp(log_pdf)

    # x_grid = np.linspace(min(x), max(x), 1000)
    # x_std = np.asarray(x).std()

    # grid = GridSearchCV(KernelDensity(),
    # {'bandwidth': np.linspace(0.1*x_std, x_std, 30)},
    # cv=min(20, len(x)//2),
    # iid=False) # 20-fold cross-validation
    # grid.fit(x[:, None])

    # kde = grid.best_estimator_
    # pdf = np.exp(kde.score_samples(x_grid[:, None]))

    # # ~ print(grid.best_params_)
    # # ~ print(max(x))

    # # ~ import matplotlib.pyplot as plt
    # # ~ fig, ax = plt.subplots()
    # # ~ ax.plot(x_grid, pdf, linewidth=3, alpha=0.5)
    # # ~ ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, density=True)

    # # ~ plt.show()
