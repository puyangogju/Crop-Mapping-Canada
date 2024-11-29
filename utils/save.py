import numpy
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio
from PIL import ImageColor
import os
import numpy as np


results_path = Path('results')

munich_colors = [
    (1, 'Beet', ImageColor.getcolor('#000000', 'RGB')),
    (2, 'Oat', ImageColor.getcolor('#4E4D4B', 'RGB')),
    (3, 'Meadow', ImageColor.getcolor('#015293', 'RGB')),
    (4, 'Rape', ImageColor.getcolor('#98C6EA', 'RGB')),
    (6, 'Spelt', ImageColor.getcolor('#DAD8CB', 'RGB')),
    (9, 'Peas', ImageColor.getcolor('#69075A', 'RGB')),
    (13, 'Wheat', ImageColor.getcolor('#689A1D', 'RGB')),
    (16, 'S.Barley', ImageColor.getcolor('#D74C13', 'RGB')),
    (17, 'Maize', ImageColor.getcolor('#C5481C', 'RGB'))
]

lombardia_colors = [
    (1, 'Cereals', ImageColor.getcolor('#FF0000', 'RGB')),
    (2, 'Woods', ImageColor.getcolor('#00FF00', 'RGB')),
    (3, 'Forage', ImageColor.getcolor('#FEA500', 'RGB')),
    (4, 'Corn', ImageColor.getcolor('#FFFF00', 'RGB')),
    (5, 'Rice', ImageColor.getcolor('#00FFFF', 'RGB')),
    (6, 'Unk. Crop', ImageColor.getcolor('#BEBEBE', 'RGB')),
    (7, 'No agric', ImageColor.getcolor('#000000', 'RGB'))
]


def apply_cmap(x, cmap):
    y = np.full(shape=(x.shape[0], x.shape[1], 3), dtype = np.uint8, fill_value=255)

    if cmap == 'munich':
        colors = munich_colors
    elif cmap == 'lombardia':
        colors = lombardia_colors

    for color in colors:
        y[x == color[0], :] = color[2]

    return y


def save_plot(pred, target, filename, model, cmap):
    plots_path =  results_path / model / 'plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    pred[target == 0] = 0
    pred = apply_cmap(pred, cmap)
    target = apply_cmap(target, cmap)

    diff = numpy.zeros(shape=(target.shape[0], target.shape[1]))
    comp = (target != pred).any(axis=2)
    diff[comp] = 1

    axs[0].imshow(pred)
    axs[1].imshow(target)
    axs[2].imshow(diff, cmap='gray', vmin=0, vmax=1)

    fig.tight_layout()

    if len(filename.split(os.sep)) == 2:
        filename = 'munich/' + filename

    out_path = plots_path / f'{filename}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ax in axs:
        ax.title.set_size(32)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(out_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def save_tiff(pred, target, filename, data_dir, model):
    preds_path =  results_path / model / 'preds'
    gt_path = results_path / model / 'truths'
    preds_path.mkdir(parents=True, exist_ok=True)
    gt_path.mkdir(parents=True, exist_ok=True)

    label_filename = data_dir / filename / 'y.tif'

    if len(filename.split(os.sep)) == 2:
        filename = 'munich/' + filename

    pred_out_path = preds_path / f'{filename}.tiff'
    pred_out_path.parent.mkdir(parents=True, exist_ok=True)
    target_out_path = gt_path / f'{filename}.tiff'
    target_out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(label_filename) as src:
        profile = src.profile
        profile['dtype'] = 'uint8'
        profile['nodata'] = 255

        with rasterio.Env():
            with rasterio.open(pred_out_path, 'w', **profile) as dst:
                dst.write(pred.astype(rasterio.uint8), 1)
            with rasterio.open(target_out_path, 'w', **profile) as dst:
                dst.write(target.astype(rasterio.uint8), 1)



def save_merged_patches(model):
    preds_path =  results_path / model / 'preds'
    gt_path = results_path / model / 'truths'
    merged_path = results_path / model / 'merged'
    merged_path.mkdir(parents=True, exist_ok=True)

    # for preds and gt patches
    for x in [preds_path, gt_path]:
        # for each area
        for area_dir in x.iterdir():
            # for each year
            for area_year_dir in area_dir.iterdir():
                parts = str(area_year_dir).split(os.sep)
                out_path = merged_path / f'{parts[-3]}_{parts[-2]}_{parts[-1]}.tiff'
                out_path.parent.mkdir(parents=True, exist_ok=True)

                command = "gdal_merge.py -a_nodata 255 -init 255 -co COMPRESS=LZW -o {} `find {} -name {}`".format(out_path, Path(area_year_dir), '*.tiff')
                os.system(command)