import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import cm
import colorcet as cc
import numpy as np
import holoviews as hv

MICROSCOPY_CMAPS = [
    'gray',
    colors.LinearSegmentedColormap.from_list('red', [(0, 0, 0, 1),
                                                     (1, 0, 0, 1)],
                                             N=256),
    colors.LinearSegmentedColormap.from_list('green', [(0, 0, 0, 1),
                                                       (0, 1, 0, 1)],
                                             N=256),
    colors.LinearSegmentedColormap.from_list('blue', [(0, 0, 0, 1),
                                                      (0, 0, 1, 1)],
                                             N=256),
    colors.LinearSegmentedColormap.from_list('cyan', [(0, 0, 0, 1),
                                                      (0, 1, 1, 1)],
                                             N=256),
    colors.LinearSegmentedColormap.from_list('magenta', [(0, 0, 0, 1),
                                                         (1, 0, 1, 1)],
                                             N=256),
    colors.LinearSegmentedColormap.from_list('yellow', [(0, 0, 0, 1),
                                                        (1, 1, 0, 1)],
                                             N=256),

    # TODO adjust definition of "hot" cmaps to match ImageJ (points are not equidistant)
    colors.LinearSegmentedColormap.from_list('cyan_hot', [(0, (0, 0, 0, 1)),
                                                          (0.7, (0, 1, 1, 1)),
                                                          (1, (1, 1, 1, 1))],
                                             N=256),
    colors.LinearSegmentedColormap.from_list('magenta_hot',
                                             [(0, (0, 0, 0, 1)),
                                              (0.7, (1, 0, 1, 1)),
                                              (1, (1, 1, 1, 1))],
                                             N=256),
    colors.LinearSegmentedColormap.from_list('yellow_hot',
                                             [(0, (0, 0, 0, 1)),
                                              (0.7, (1, 1, 0, 1)),
                                              (1, (1, 1, 1, 1))],
                                             N=256),
]

for cmap in MICROSCOPY_CMAPS:
    if not isinstance(cmap, str):
        cm.register_cmap(name=cmap.name, cmap=cmap)

available_cmaps = {
    c if isinstance(c, str) else c.name: c
    for c in MICROSCOPY_CMAPS
}

# repeat colormap to handle unint16 values
# needed to handle non continuous labels because colormap is stretched (and not cycled)
blk_glasbey_hv_16bit = ['#000000'] + (cc.b_glasbey_hv * 256)[:-1]
available_cmaps['blk_glasbey_hv_16bit'] = blk_glasbey_hv_16bit

glasbey_hv_16bit = cc.b_glasbey_hv * 256
available_cmaps['glasbey_hv_16bit'] = glasbey_hv_16bit

blk_glasbey_hv = ['#000000'] + cc.b_glasbey_hv[:-1]
available_cmaps['blk_glasbey_hv'] = blk_glasbey_hv

glasbey_hv = cc.b_glasbey_hv
available_cmaps['glasbey_hv'] = glasbey_hv

# register glasbey_hv_16bit with matplotlib for reference, however colormap end up covnerted to 8 bit if access by name with holoviews
# --> directly use lsit of color when plotting with holoviews
cm.register_cmap(name='blk_glasbey_hv_16bit',
                 cmap=colors.ListedColormap(blk_glasbey_hv_16bit,
                                            name='blk_glasbey_hv_16bit'))
cm.register_cmap(name='glasbey_hv_16bit',
                 cmap=colors.ListedColormap(blk_glasbey_hv_16bit,
                                            name='glasbey_hv_16bit'))
cm.register_cmap(name='glasbey_hv',
                 cmap=colors.ListedColormap(glasbey_hv, name='glasbey_hv'))
cm.register_cmap(name='blk_glasbey_hv',
                 cmap=colors.ListedColormap(blk_glasbey_hv,
                                            name='blk_glasbey_hv'))

mesh_cmaps = [
    colors.LinearSegmentedColormap.from_list('clipped_plasma',
                                             plt.get_cmap('plasma')(
                                                 range(230)),
                                             N=256),
    colors.LinearSegmentedColormap.from_list('clipped_plasma_r',
                                             plt.get_cmap('plasma_r')(range(
                                                 15, 256)),
                                             N=256)
]

for cmap in mesh_cmaps:
    if not isinstance(cmap, str):
        cm.register_cmap(name=cmap.name, cmap=cmap)

clipped_plasma_r = plt.get_cmap('clipped_plasma_r')
clipped_plasma = plt.get_cmap('clipped_plasma')


def plot_colorbar(cmap,
                  bounds,
                  label=None,
                  orientation='vertical',
                  backend='matplotlib'):
    if label is None:
        label = ''

    cbar_img = np.linspace(0, 1, 256)
    vmin, vmax = bounds

    opts_kwargs = {
        'cmap': cmap,
        'show_frame': False,
        'framewise': True,
        'axiswise': True
    }

    if orientation == 'vertical':
        cbar_img = cbar_img[:, None]
        bounds = (0, vmin, 1, vmax)

        opts_kwargs['xaxis'] = None
        opts_kwargs['yaxis'] = 'right'
        opts_kwargs['ylabel'] = ''
        opts_kwargs['yticks'] = 5

    elif orientation == 'horizontal':
        cbar_img = cbar_img[None]
        bounds = (vmin, 0, vmax, 1)

        opts_kwargs['xaxis'] = 'bottom'
        opts_kwargs['yaxis'] = None
        opts_kwargs['xlabel'] = ''
        opts_kwargs['xticks'] = 5

    else:
        raise ValueError(
            'orientation not recognized. expects vertical|horizontal, got {}'.
            format(orientation))

    if backend == 'matplotlib':
        opts_kwargs['sublabel_size'] = 0
        opts_kwargs['fontsize'] = {'title': 14}
        if orientation == 'vertical':
            opts_kwargs['aspect'] = 0.02
        else:
            opts_kwargs['aspect'] = 1 / 0.02

    elif backend == 'bokeh':
        opts_kwargs['fontsize'] = {'title': 8}
        opts_kwargs['toolbar'] = None
        if orientation == 'vertical':
            opts_kwargs['frame_width'] = 15
        else:
            opts_kwargs['frame_height'] = 15

    else:
        raise ValueError(
            'backend not supported. expects matplotlib|bokeh, got {}'.format(
                backend))

    return hv.Image(cbar_img, label=label, bounds=bounds).opts(**opts_kwargs)
