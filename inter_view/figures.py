import itertools
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from inter_view.utils import make_composite


class CompositeFigureMaker():
    '''Creates a colof figure with legends and scale bar from a list of greyscale channels.
    
    by default, attempts to place the scale bar and legend in image regions with the lowest 
    intensities and the "standard" scale bar the closest to 1/6th of image width.
    '''

    regions = {
        (0, 0): 'upper left',
        (0, 1): 'upper center',
        (0, 2): 'upper right',
        (1, 2): 'center right',
        (2, 2): 'lower right',
        (2, 1): 'lower center',
        (2, 0): 'lower left',
        (1, 0): 'center left'
    }

    phy_bar_lengths = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])

    def __init__(self,
                 fig_size=(12, 12),
                 dpi=72,
                 font_size=18,
                 scale=None,
                 show_legend=False,
                 bar_lengths=None,
                 scale_bar_pos=None,
                 legend_pos=None):
        self.fig_size = fig_size
        self.dpi = dpi
        self.font_size = font_size
        self.scale = scale
        self.show_legend = show_legend
        if bar_lengths is not None:
            self.phy_bar_lengths = np.array(bar_lengths)

        if (scale_bar_pos is not None) != (legend_pos is not None):
            raise ValueError(
                'neither or both scale bar and legend positions should be specified'
            )

        self.scale_bar_pos = scale_bar_pos
        self.legend_pos = legend_pos

    def __call__(self, imgs, cmaps, intensity_bounds, labels):

        imgs = [
            rescale_intensity(img, in_range=b, out_range=np.uint8)
            for img, b in zip(imgs, intensity_bounds)
        ]
        rgb_img = make_composite(imgs, cmaps)

        fig, ax = plt.subplots(1,
                               1,
                               figsize=self.fig_size,
                               dpi=self.dpi,
                               frameon=False)
        ax.imshow(rgb_img, aspect=1)
        ax.set_axis_off()
        ax.axis('off')

        if self.scale_bar_pos is not None:
            legend_pos = self.legend_pos
            scale_bar_pos = self.scale_bar_pos
        else:
            legend_pos, scale_bar_pos = self.find_best_overlay_pos(rgb_img)

        if self.show_legend:
            self.draw_label(fig, ax, labels, cmaps, legend_pos)
        if self.scale:
            self.draw_scalebar(ax, rgb_img.shape[:2], scale_bar_pos)

        return fig

    def draw_label(self, fig, ax, labels, cmaps, legend_pos):

        dummy_lines = [
            Line2D([0], [0], lw=4),
            Line2D([0], [0], lw=4),
            Line2D([0], [0], lw=4)
        ]

        leg = ax.legend(dummy_lines,
                        labels,
                        handlelength=0,
                        loc=legend_pos,
                        fontsize=self.font_size,
                        frameon=False,
                        labelspacing=0.0)

        for cmap, text in zip(cmaps, leg.get_texts()):
            text.set_color(plt.get_cmap(cmap)(0.9))

    def draw_scalebar(self, ax, img_shape, scale_bar_pos):
        # find closest standard size to bar length ~= 1/6 fig_width
        target_phy_bar_length = self.scale * img_shape[1] / 6
        diff = np.abs(self.phy_bar_lengths - target_phy_bar_length)
        phy_bar_length = self.phy_bar_lengths[np.argmin(diff)]
        px_bar_length = phy_bar_length / self.scale

        # scalebar https://stackoverflow.com/questions/39786714/how-to-insert-scale-bar-in-a-map-in-matplotlib
        scalebar = AnchoredSizeBar(ax.transData,
                                   px_bar_length,
                                   '{} μm'.format(phy_bar_length),
                                   scale_bar_pos,
                                   pad=0.5,
                                   color='white',
                                   frameon=False,
                                   size_vertical=min(10, px_bar_length / 10),
                                   fontproperties=fm.FontProperties(size=18))

        ax.add_artist(scalebar)

    def get_tiles(self, shape, nx_tiles, ny_tiles):
        '''return tiles slice indexing with the specified number of tiles'''
        tiles = []

        xt_size, yt_size = np.array(shape) / (nx_tiles, ny_tiles)
        for xt, yt in itertools.product(range(nx_tiles), range(ny_tiles)):
            xsl = slice(int(round(xt * xt_size)), int(round(
                (xt + 1) * xt_size)))
            ysl = slice(int(round(yt * yt_size)), int(round(
                (yt + 1) * yt_size)))

            tiles.append(((xt, yt), (xsl, ysl)))

        return tiles

    def find_best_overlay_pos(self, img):
        '''finds empty regions to draw legend and scale bar'''

        tiles = self.get_tiles(img.shape[:2], 3, 3)
        # remove middle tile
        tiles.pop(4)
        tiles_mean_intensity = [(t_id, img[loc].mean()) for t_id, loc in tiles]
        tiles_mean_intensity.sort(key=lambda t: t[1])

        legend_pos = self.regions[tiles_mean_intensity[0][0]]
        scale_bar_pos = self.regions[tiles_mean_intensity[1][0]]

        return legend_pos, scale_bar_pos
