""" Compute metrics on test set using config.py settings. """
import numpy as np
import os
import h5py as h5
import yaml
from utils.evaluate import evaluate, make_figure
from models import SFANet, SFANetUnet
from utils.preprocess import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import pandas as pd

import sys, importlib
config_name = next((sys.argv[i + 1] for i, v in enumerate(sys.argv) if v == '--config' and i + 1 < len(sys.argv)), 'config')
config = importlib.import_module(config_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config', help='Config module name (already processed)')
    parser.add_argument('--peak_value', type=float, default=config.peak_value, help='num epochs')
    args = parser.parse_args()
    
    peak_value_str = str(int(args.peak_value * 100)) if args.peak_value is not None else 'best'


    
    # Load parameters
    params_path = os.path.join(config.log_directory, 'params_detection.yaml')
    if os.path.exists(params_path) and args.peak_value is None:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            # mode = params['mode']
            min_distance = params['min_distance']
            threshold_abs = params['threshold_abs']
            threshold_rel = None
            # threshold_rel = params['threshold_rel'] if mode == 'rel' else None
    else:
        print(f'Warning: params_detection.yaml missing -- using default params')
        min_distance = config.min_distance
        threshold_abs = args.peak_value
        threshold_rel = None
    
    geo_mode = getattr(config, 'geo_mode', 'unit')  # 'unit' | 'fourier' | 'none'
    freqs    = getattr(config, 'geo_fourier_freqs', (1,2,4,8))

    if geo_mode in (None, 'none'):
        geo_dim = 0
    elif geo_mode == 'unit':
        geo_dim = 3
    elif geo_mode == 'fourier':
        geo_dim = 4 * len(freqs)   # 2 coords * 2 (sin/cos) * len(freqs) = 16 if (1,2,4,8)
    else:
        raise ValueError(f"Unsupported geo_mode: {geo_mode}")

    # Load HDF5 test data
    f = h5.File(config.prepared_data_path, 'r')
    images = f['test/images'][:]
    gts    = f['test/gt'][:]

    # Optional: read geos if model expects them
    geo_ds = f'test/geo_{geo_mode}'
    if geo_mode != 'none' and geo_ds in f:
        test_geos = f[geo_ds][:]
        # early tripwire for shape mismatches
        assert test_geos.shape[0] == images.shape[0], "geo/test count != images/test count"
        assert test_geos.shape[1] == geo_dim, f"geo width {test_geos.shape[1]} != expected {geo_dim}"
    else:
        test_geos = None

    bands = f.attrs['bands']
    preprocess = eval(f'preprocess_{bands}')
    
    # Build model (same options you used in training)
    if config.model == 'vgg':
        model_builder = SFANet
    elif config.model == 'unet':
        model_builder = SFANetUnet
    else:
        raise ValueError(f"Unsupported model type: {config.model}")

    # Important: pass geo_mode/geo_dim so the input signature matches the weights
    train_model, model = model_builder.build_model(
        images.shape[1:], preprocess_fn=preprocess,
        geo_mode=geo_mode, geo_dim=(None if geo_dim==0 else geo_dim),
        film_hidden=getattr(config,'geo_film_hidden',64),
        film_dropout=getattr(config,'geo_film_dropout',0.1),
        backbone=getattr(config, 'backbone', None),
        encoder_weights=getattr(config, 'pretrain_w', None)
    )

    weights_path = os.path.join(config.log_directory, 'best.weights.h5') #changed BACK to original from the TEMPORARY FIX of config.init_weights (used when skipping training but doing testing)
    train_model.load_weights(weights_path)

    print('----- Getting predictions from trained model -----')
    if test_geos is None:
        preds = model.predict(images, verbose=True, batch_size=1)[..., 0]
    else:
        preds = model.predict([images, test_geos], verbose=True, batch_size=1)[..., 0]

    print('----- Calculating metrics -----')
    print(f'Using minimum distance: {min_distance} and threshold: {threshold_abs}')
    results = evaluate(
        gts=gts,
        preds=preds,
        method=config.peak_method,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        max_distance=10,  # default max distance (can be parameterized if needed)
        h_value=config.h_max_value,
        return_locs=True
    )
    
    # ---------- Per-image evaluation ----------
    def _get_image_names(h5file, n_imgs):
        for key in ['test/filenames', 'test/paths', 'test/image_names', 'test/ids', 'test/tiles', 'test/names']:
            if key in h5file:
                arr = h5file[key][:]
                arr = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]
                return arr[:n_imgs] + [f"test_{i:05d}" for i in range(len(arr), n_imgs)]
        # fallback if none found
        return [f"test_{i:05d}" for i in range(n_imgs)]
  
    n_imgs = images.shape[0]
    image_names = _get_image_names(f, n_imgs)
    valid_pixel_counts = np.count_nonzero(images[..., 0], axis=(1, 2))
    resolution_m = config.reproject_resolution  # meters per pixel 
    area_per_image_ha = (valid_pixel_counts * (resolution_m ** 2)) / 10_000
    
    # Build per-image tallies with precision/recall/F1
    rows = []
    for i in range(n_imgs):
        tp = len(results['tp_locs'][i]) if i < len(results['tp_locs']) else 0
        fp = len(results['fp_locs'][i]) if i < len(results['fp_locs']) else 0
        fn = len(results['fn_locs'][i]) if i < len(results['fn_locs']) else 0
        gt = len(results['gt_locs'][i]) if i < len(results['gt_locs']) else 0
        pred = tp + fp

        # Calculate precision, recall, and F1
        if tp == 0 and fp == 0 and fn == 0:
            precision, recall, f1 = 1.0, 1.0, 1.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            

        rows.append({
            'test_image': image_names[i],
            'area_ha':  area_per_image_ha[i], 
            'gt_count': int(gt),
            'pred_count': int(pred),
            'tp_count': int(tp),
            'fp_count': int(fp),
            'fn_count': int(fn),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        })

    per_image_df = pd.DataFrame(
        rows,
        columns=['test_image', 'area_ha', 'gt_count', 'pred_count', 'tp_count', 'fp_count', 'fn_count', 'precision', 'recall', 'f1']
    )
    per_image_df.insert(1, 'model', config.model_name)
    per_image_df.insert(2, 'peak_value', threshold_abs)
    os.makedirs(config.log_directory, exist_ok=True)
    per_image_csv = os.path.join(config.log_directory, f'results_per_image_{peak_value_str}.csv')
    per_image_df.to_csv(per_image_csv, index=False)
    print(f'Per-image metrics saved to {per_image_csv}')

    
    
    ##### Save overall metrics to text file
    os.makedirs(config.log_directory, exist_ok=True)
    with open(os.path.join(config.log_directory, f'results_{peak_value_str}.txt'), 'w') as f:
        f.write(f'model: {config.model_name}\n')
        f.write(f'peak_value: {threshold_abs}\n')
        f.write('precision: ' + str(results['precision']) + '\n')
        f.write('recall: ' + str(results['recall']) + '\n')
        f.write('fscore: ' + str(results['fscore']) + '\n')
        f.write('rmse [px]: ' + str(results['rmse']) + '\n')

    print('------- Results ---------')
    print('Precision:', results['precision'])
    print('Recall:', results['recall'])
    print('F-score:', results['fscore'])
    print('RMSE [px]:', results['rmse'])

    # Save evaluation visualizations as multipage PDF
    vis_path = os.path.join(config.log_directory, 'figure.pdf')
    with PdfPages(vis_path) as pdf:
        num_images = images.shape[0]
        batch_size = 12  # 3 rows × 4 columns
        for start in range(0, num_images, batch_size):
            end = min(start + batch_size, num_images)
            batch_imgs = images[start:end, ..., [3, 0, 1]]
            batch_results = {
                'tp_locs': results['tp_locs'][start:end],
                'tp_gt_locs': results['tp_gt_locs'][start:end],
                'fp_locs': results['fp_locs'][start:end],
                'fn_locs': results['fn_locs'][start:end],
                'gt_locs': results['gt_locs'][start:end],
            }
            fig = make_figure(batch_imgs, batch_results, num_cols=4)
            fig.set_size_inches(12, 9)
    
            # --- Add image names as subplot titles ---
            axes = fig.get_axes()
            for idx, ax in enumerate(axes):
                img_idx = start + idx
                if img_idx < len(image_names):
                    ax.set_title(image_names[img_idx], fontsize=8)
    
            pdf.savefig(fig)
            plt.close(fig)
    print(f'Visualization saved to {vis_path}')


    # Generate scatter plot of predicted vs actual dead tree counts per image (normalized per hectare)

    # --- Calculate per-hectare densities ---
    gt_counts = np.array([len(pts) for pts in results['gt_locs']], dtype=int)
    pred_counts = np.array([len(tp) + len(fp) for tp, fp in zip(results['tp_locs'], results['fp_locs'])], dtype=int)

    # Compute densities
    gt_counts_ha = gt_counts / area_per_image_ha
    pred_counts_ha = pred_counts / area_per_image_ha
    
    # Calculate RMSE and R^2
    rmse = np.sqrt(mean_squared_error(gt_counts_ha, pred_counts_ha))
    r2 = r2_score(gt_counts_ha, pred_counts_ha)

    # Scatter plot with annotations
    plt.figure(figsize=(10, 8))
    plt.scatter(gt_counts_ha, pred_counts_ha, c='blue', edgecolors='k', alpha=0.6)
    plt.plot([min(gt_counts_ha), max(gt_counts_ha)], [min(gt_counts_ha), max(gt_counts_ha)], 'r--', label='y = x')
    plt.xlabel('Actual Dead Trees per Hectare')
    plt.ylabel('Predicted Dead Trees per Hectare')
    plt.title('Predicted vs Actual Dead Tree Density')
    plt.legend()
    plt.grid(True)

    # Add metrics as text
    text_str = f"RMSE = {rmse:.2f}\n$R^2$ = {r2:.2f}"
    plt.gca().text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    scatter_path = os.path.join(config.log_directory, f'scatter_{peak_value_str}.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f'Scatter plot saved to {scatter_path}')


if __name__ == '__main__':
    main()
