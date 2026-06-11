import argparse
import os
import h5py
import numpy as np
import random
import tqdm
import rasterio
from scipy.ndimage import distance_transform_edt
from utils.geo import encode_geo, centroid_wgs84_from_src

import sys, importlib
config_name = next((sys.argv[i + 1] for i, v in enumerate(sys.argv) if v == '--config' and i + 1 < len(sys.argv)), 'config')
config = importlib.import_module(config_name)

def load_data(dataset_path, names, sigma, bands, geo_freqs=(1,2,4,8)):
    data = []
    for name in tqdm.tqdm(names, desc="Loading data"):
        image = None; lat = lon = None
        for suffix in ['.tif', '.tiff', '.png']:
            image_path = os.path.join(dataset_path, 'images', name + suffix)
            if os.path.exists(image_path):
                with rasterio.open(image_path) as src:
                    arr = src.read()  # (bands, H, W)
                    image = np.transpose(arr, (1, 2, 0))  # (H, W, bands)
                    lat, lon = centroid_wgs84_from_src(src) # compute centroid in WGS84 and encode
                if suffix == '.png' or bands == 'RGB':
                    image = image[..., :3]
                break
        if image is None:
            raise RuntimeError(f"Could not find image for {name}")

        # labels → confidence/attention
        csv_path = os.path.join(dataset_path, 'csv', name + '.csv')
        if os.path.exists(csv_path):
            points = np.loadtxt(csv_path, delimiter=',', skiprows=1).astype('int')
            if points.ndim == 1: points = points[None, :]
            gt = np.zeros(image.shape[:2], dtype='float32')
            gt[points[:, 1], points[:, 0]] = 1
            distance = distance_transform_edt(1 - gt).astype('float32')
            confidence = np.exp(-distance**2 / (2 * sigma**2))
        else:
            gt = np.zeros(image.shape[:2], dtype='float32')
            confidence = np.zeros(image.shape[:2], dtype='float32')

        confidence = confidence[..., None]
        attention  = (confidence > 0.001).astype('float32')

        # --- ALWAYS compute both geo encodings ---
        geo_unit    = encode_geo(lat, lon, mode="unit").astype('float32')                # (3,)
        geo_fourier = encode_geo(lat, lon, mode="fourier", freqs=geo_freqs).astype('float32')  # (16,)

        rec = {
            'name': name,
            'image': image,
            'gt': gt,
            'confidence': confidence,
            'attention': attention,
            'geo_unit': geo_unit, # (N, 3)
            'geo_fourier': geo_fourier, # (N, 16)
        }
        data.append(rec)
    return data

def count_labels_for_names(dataset_path, names):
    """Return total number of tree points across csv/<name>.csv for a list of names."""
    total = 0
    for n in names:
        p = os.path.join(dataset_path, 'csv', n + '.csv')
        if not os.path.exists(p):
            continue
        try:
            # robust line counting: header + rows; handle empty files gracefully
            with open(p, 'r') as f:
                lines = sum(1 for _ in f)
            total += max(0, lines - 1)  # subtract header
        except Exception:
            # as a fallback, try numpy
            try:
                arr = np.loadtxt(p, delimiter=',', skiprows=1)
                if arr.ndim == 1 and arr.size > 0:
                    total += 1
                elif arr.ndim >= 1:
                    total += int(arr.shape[0])
            except Exception:
                pass
    return total

def augment_images(images):
    augmented = np.concatenate((images,
                                 np.rot90(images, k=1, axes=(1, 2)),
                                 np.rot90(images, k=2, axes=(1, 2)),
                                 np.rot90(images, k=3, axes=(1, 2))))
    augmented = np.concatenate((augmented, np.flip(augmented, axis=-2)))
    return augmented

def read_names(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def sample_fraction(names, frac, seed, split=None, save_dir=None):
    if frac < 1.0:
        random.seed(seed)
        sample_size = max(1, int(len(names) * frac))
        sampled = random.sample(names, sample_size)
        if split and save_dir:
            out_path = os.path.join(save_dir, f'{split}_frac{frac:.2f}.txt')
            with open(out_path, 'w') as f:
                f.write('\n'.join(sampled))
        return sampled
    return names


def add_data_to_h5(f, data, split, augment=False):
    if len(data) == 0:
        return

    names      = [d['name'] for d in data]
    images     = np.stack([d['image'] for d in data], axis=0)
    gt         = np.stack([d['gt'] for d in data], axis=0)
    confidence = np.stack([d['confidence'] for d in data], axis=0)
    attention  = np.stack([d['attention'] for d in data], axis=0)

    geo_unit    = np.stack([d['geo_unit']    for d in data], axis=0).astype('float32')
    geo_fourier = np.stack([d['geo_fourier'] for d in data], axis=0).astype('float32')

    if augment:
        names      = np.repeat(names, 8)
        images     = augment_images(images)
        gt         = augment_images(gt)
        confidence = augment_images(confidence)
        attention  = augment_images(attention)
        geo_unit    = np.repeat(geo_unit,    8, axis=0)
        geo_fourier = np.repeat(geo_fourier, 8, axis=0)

    f.create_dataset(f'{split}/names',      data=np.array(names, dtype=h5py.string_dtype()))
    f.create_dataset(f'{split}/images',     data=images)
    f.create_dataset(f'{split}/gt',         data=gt)
    f.create_dataset(f'{split}/confidence', data=confidence)
    f.create_dataset(f'{split}/attention',  data=attention)

    f.create_dataset(f'{split}/geo_unit',    data=geo_unit) 
    f.create_dataset(f'{split}/geo_fourier', data=geo_fourier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=config.output_directory, help='path to dataset')
    parser.add_argument('--output', default=config.prepared_data_path, help='output path for .hdf5 file')
    parser.add_argument('--data_frac', type=float, default=config.data_frac, help='Fraction of data to use (0–1, 1=full)')
    parser.add_argument('--prepared_split', default=config.prepared_split, help='Optional subfolder for split (e.g., split1)')
    parser.add_argument('--augment', action='store_true', help='apply augmentation')
    parser.add_argument('--sigma', type=float, default=config.sigma, help='Gaussian kernel size in pixels')
    parser.add_argument('--bands', default=config.bands, help='input raster bands (RGB or RGBN)')
    parser.add_argument('--seed', type=int, default=config.seed, help='Random seed for sampling in small mode')
    parser.add_argument('--config', type=str, default='config', help='Config module name (already processed)')

    args = parser.parse_args()

    split_dir = os.path.join(args.dataset, args.prepared_split) 
    train_names = sample_fraction(read_names(os.path.join(split_dir, 'train.txt')), args.data_frac, args.seed, split='train', save_dir=split_dir)
    val_names   = sample_fraction(read_names(os.path.join(split_dir, 'val.txt')),   args.data_frac, args.seed, split='val',   save_dir=split_dir)
    test_names = read_names(os.path.join(split_dir, 'test.txt'))

    train_data = load_data(args.dataset, train_names, args.sigma, args.bands,
                           geo_freqs=getattr(config, 'geo_fourier_freqs', (1,2,4,8)))
    val_data   = load_data(args.dataset, val_names, args.sigma, args.bands,
                           geo_freqs=getattr(config, 'geo_fourier_freqs', (1,2,4,8)))
    test_data  = load_data(args.dataset, test_names, args.sigma, args.bands,
                           geo_freqs=getattr(config, 'geo_fourier_freqs', (1,2,4,8)))
    
    with h5py.File(args.output, 'w') as f:
        add_data_to_h5(f, train_data, 'train', augment=args.augment)
        add_data_to_h5(f, val_data, 'val')
        add_data_to_h5(f, test_data, 'test')
        f.attrs['bands'] = args.bands

    print(f"✓ HDF5 file saved to: {args.output}")
    
    
    f_attrs = {
    'bands': args.bands,
    'data_frac': float(args.data_frac)
    }
    with h5py.File(args.output, 'a') as f:
        for k, v in f_attrs.items():
            f.attrs[k] = v
    
    summary_path = os.path.splitext(args.output)[0] + ".csv"


    counts = {
        'train': {'images': len(train_names), 'trees': count_labels_for_names(args.dataset, train_names)},
        'val':   {'images': len(val_names),   'trees': count_labels_for_names(args.dataset, val_names)},
        'test':  {'images': len(test_names),  'trees': count_labels_for_names(args.dataset, test_names)},
    }
    tot_images = sum(v['images'] for v in counts.values())
    tot_trees  = sum(v['trees']  for v in counts.values())

    with open(summary_path, 'w') as sf:
        sf.write(f"Prepared dataset: {config.prepared_fname}\n")
        sf.write(f"Dataset root    : {args.dataset}\n")
        sf.write(f"Split folder    : {args.prepared_split}\n")
        sf.write(f"Bands           : {args.bands}\n")
        sf.write(f"Data fraction   : {args.data_frac}\n")
        sf.write("\n")
        for split in ('train', 'val', 'test'):
            sf.write(f"[{split}]\n")
            sf.write(f"  images: {counts[split]['images']}\n")
            sf.write(f"  labels: {counts[split]['trees']}\n\n")
        sf.write("TOTAL\n")
        sf.write(f"  images: {tot_images}\n")
        sf.write(f"  labels: {tot_trees}\n")

    print(f"✓ Summary saved to: {summary_path}")

if __name__ == '__main__':
    main()
