#!/usr/bin/env python
import os, sys, yaml, json, argparse, importlib, random, time, gc
import tqdm
import numpy as np
import rasterio
import tensorflow as tf
from models import SFANet, SFANetUnet
from utils.preprocess import *
from utils.inference import run_tiled_inference
from utils.geo import encode_geo

# --- safer GPU usage: prevent TensorFlow from reserving all GPU memory ---
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def _coerce_to_list(x):
    """Accept string, comma-list, JSON list, or list; always return list[str]."""
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed]
            except Exception:
                pass
        if ',' in s:
            return [t.strip() for t in s.split(',') if t.strip()]
        return [s]
    return [str(x)]


# --- Load config dynamically ---
config_name = next(
    (sys.argv[i + 1] for i, v in enumerate(sys.argv) if v == '--config' and i + 1 < len(sys.argv)),
    'config'
)
config = importlib.import_module(config_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config', help='Config module name')
    parser.add_argument('--input', default=config.inference_data_path, help='Root input directory')
    parser.add_argument('--output', default=config.inference_out_directory, help='Root output directory')
    parser.add_argument('--saved_model_dir', default=config.saved_model_directory, help='Directory with saved models')
    parser.add_argument('--inf_model_name', default=config.inf_model_name, help='One or more model names')
    parser.add_argument('--bands', default='RGBN', help='Input bands, e.g. RGB or RGBN')
    parser.add_argument('--tile_size', type=int, default=2048, help='Tile size for inference')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap between tiles')
    parser.add_argument('--stable_min', type=float, default=0,
                        help='Minimum file age (minutes) before processing (to skip files still being written)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Normalize model list ---
    model_names = _coerce_to_list(args.inf_model_name)

    # --- Geo settings ---
    geo_mode = getattr(config, 'geo_mode', 'unit')
    freqs = getattr(config, 'geo_fourier_freqs', (1, 2, 4, 8))

    min_distance = getattr(config, 'min_distance', 3)
    threshold_abs = getattr(config, 'peak_value', 0.5)
    threshold_rel = None

    # --- Try to load params.yaml if available ---
    for mn in model_names:
        params_path = os.path.join(args.saved_model_dir, mn, 'params.yaml')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = yaml.safe_load(f)
            mode = params.get('mode', 'abs')
            min_distance = params.get('min_distance', min_distance)
            threshold_abs = params.get('threshold_abs', threshold_abs) if mode == 'abs' else None
            threshold_rel = params.get('threshold_rel', None) if mode == 'rel' else None
            break

    # --- Geo dimension ---
    if geo_mode in (None, 'none'):
        geo_dim = 0
    elif geo_mode == 'unit':
        geo_dim = 3
    elif geo_mode == 'fourier':
        geo_dim = 4 * len(freqs)
    else:
        raise ValueError(f"Unsupported geo_mode: {geo_mode}")

    # --- Build model(s) ---
    model_type = config.model
    padded_size = args.tile_size + args.overlap * 2
    preprocess = eval(f'preprocess_{args.bands}')
    input_shape = (padded_size, padded_size, len(args.bands))

    if model_type == 'vgg':
        builder = SFANet
    elif model_type == 'unet':
        builder = SFANetUnet
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    models = []
    for mn in model_names:
        log = os.path.join(args.saved_model_dir, mn)
        weights_path = os.path.join(log, 'best.weights.h5')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights: {weights_path}")

        train_model, inf_model = builder.build_model(
            input_shape,
            preprocess_fn=preprocess,
            geo_mode=geo_mode,
            geo_dim=(None if geo_dim == 0 else geo_dim),
            film_hidden=getattr(config, 'geo_film_hidden', 64),
            film_dropout=getattr(config, 'geo_film_dropout', 0.1),
            backbone=getattr(config, 'backbone', None),
            encoder_weights=getattr(config, 'pretrain_w', None)
        )
        train_model.load_weights(weights_path)
        models.append(inf_model)

    # --- Recursively collect input files ---
    all_files = []
    for root, _, files in os.walk(args.input):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff')):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"⚠️ No input TIFFs found in {args.input}")
        return

    # --- Filter out files modified within the last N minutes ---
    if args.stable_min > 0:
        now = time.time()
        stable_files = []
        for f in all_files:
            try:
                age_min = (now - os.path.getmtime(f)) / 60
                if age_min >= args.stable_min:
                    stable_files.append(f)
                else:
                    print(f"⏸️ Skipping (modified {age_min:.1f} min ago): {f}")
            except Exception as e:
                print(f"⚠️ Could not check mtime for {f}: {e}")
        all_files = stable_files
        print(f"✅ Found {len(all_files)} TIFFs older than {args.stable_min} min.")
        if not all_files:
            print("⚠️ No stable TIFFs found. Exiting.")
            return

    # --- Shuffle order for distributed runs ---
    random.seed(os.getpid() + int(time.time()))
    random.shuffle(all_files)

    # --- Output settings ---
    output_raster = getattr(config, "output_raster", True)
    output_centroids = getattr(config, "output_centroids", True)

    # --- Process each file ---
    pbar = tqdm.tqdm(total=len(all_files))
    for input_path in all_files:
        rel_dir = os.path.relpath(os.path.dirname(input_path), args.input)
        out_dir = os.path.join(args.output, rel_dir)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        out_stem = f"conf_{stem}"

        # Define raster and vector output folders
        raster_dir = out_dir + "_raster"
        vect_dir = out_dir + "_vect"
        os.makedirs(raster_dir, exist_ok=True)
        os.makedirs(vect_dir, exist_ok=True)

        raster_out = os.path.join(raster_dir, out_stem + ".tif")
        gpkg_out = os.path.join(vect_dir, out_stem + ".gpkg")

        # --- Skip logic ---
        already_done = True
        if output_raster and not os.path.exists(raster_out):
            already_done = False
        if output_centroids and not os.path.exists(gpkg_out):
            already_done = False

        if already_done:
            print(f"⏩ Skipping already processed file: {stem}")
            pbar.update(1)
            continue

        # --- Run inference ---
        try:
            print(f"🛰️ Processing: {input_path}")
            run_tiled_inference(
                models=models,
                input_path=input_path,
                output_path=raster_out,
                min_distance=min_distance,
                threshold_abs=threshold_abs,
                threshold_rel=threshold_rel,
                tile_size=args.tile_size,
                overlap=args.overlap,
            )
        except Exception as e:
            print(f"❌ Error on {input_path}: {e}")
            continue
        finally:
            gc.collect()
            tf.keras.backend.clear_session()

        pbar.update(1)

    pbar.close()
    print("✅ All processing complete!")


if __name__ == '__main__':
    main()
