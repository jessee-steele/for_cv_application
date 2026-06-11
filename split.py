import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import wasserstein_distance

import argparse
import sys, importlib

config_name = next((sys.argv[i + 1] for i, v in enumerate(sys.argv) if v == '--config' and i + 1 < len(sys.argv)), 'config')
config = importlib.import_module(config_name)

def strip_extension(filename):
    return os.path.splitext(filename)[0]

def save_split(split_dir, train_names, val_names, test_names):
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, 'train.txt'), 'w') as f:
        f.writelines(f"{name}\n" for name in train_names)
    with open(os.path.join(split_dir, 'val.txt'), 'w') as f:
        f.writelines(f"{name}\n" for name in val_names)
    with open(os.path.join(split_dir, 'test.txt'), 'w') as f:
        f.writelines(f"{name}\n" for name in test_names)

def optimal_wasserstein_splits(df, train_ratio, n_splits, seed=42, trials=50):
    np.random.seed(seed)
    results = []

    names = df['name'].tolist()
    values = df['count'].values
    n_total = len(df)
    n_train = int(n_total * train_ratio)

    for _ in range(trials):
        train_idx = np.random.choice(n_total, size=n_train, replace=False)
        val_idx = list(set(range(n_total)) - set(train_idx))

        train_vals = values[train_idx]
        val_vals = values[val_idx]
        dist = wasserstein_distance(train_vals, val_vals)
        results.append((dist, train_idx, val_idx))

    best_splits = sorted(results, key=lambda x: x[0])[:n_splits]

    return [(df.iloc[train], df.iloc[val]) for _, train, val in best_splits]

def run_split(args):
    image_dir = os.path.join(args.dataset, 'images')
    csv_dir = os.path.join(args.dataset, 'csv')
    all_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff', '.png'))]
    all_names = sorted(set(strip_extension(f) for f in all_files))

    label_counts = {}
    for name in all_names:
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                label_counts[name] = sum(1 for line in f) - 1
        else:
            label_counts[name] = 0

    df = pd.DataFrame({'name': list(label_counts.keys()), 'count': list(label_counts.values())})
    bins = [0, 5, np.inf]
    df['bin'] = pd.cut(df['count'], bins=bins, labels=False, include_lowest=True)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Split test set
    existing_names_set = set(all_names)  # names present in images/ (by base name)

    if hasattr(config, 'test_fname') and os.path.isfile(config.test_fname):
        with open(config.test_fname, 'r') as f:
            raw_test_names = [line.strip() for line in f if line.strip()]

        # Keep only names that truly exist in images/
        test_names, dropped = _filter_existing_test_names(raw_test_names, existing_names_set)

        if dropped:
            print(f"⚠️ {len(dropped)} test names in {config.test_fname} not found in images/. They will be ignored.")
            # # Optional: write an audit list next to the provided file
            # miss_path = os.path.join(os.path.dirname(config.test_fname), "test_missing_in_images.txt")
            # with open(miss_path, "w") as mf:
            #     mf.write("\n".join(dropped))
            # print(f"   → Saved missing list to: {miss_path}")

        if len(test_names) == 0:
            # Fallback to automatic sampling if nothing valid remains
            print("⚠️ No valid test names left after filtering. Generating test set automatically.")
            test_df = df.sample(n=args.test_size, random_state=args.seed)
        else:
            print(f"📝 Test file found. Using {len(test_names)} of {len(raw_test_names)} names from {config.test_fname}.")
            test_df = pd.DataFrame({'name': test_names})

        remaining_df = df[~df['name'].isin(test_df['name'])].reset_index(drop=True)

    else:
        test_df = df.sample(n=args.test_size, random_state=args.seed)
        remaining_df = df[~df['name'].isin(test_df['name'])].reset_index(drop=True)
        print("⚠️ Test file not found. Generating test file automatically.")


    if args.split_method == "wasserstein":
        print(f"💧 Performing Wasserstein-based splitting with {args.split} folds...")
        splits = optimal_wasserstein_splits(remaining_df, args.train_val_ratio, args.split, seed=args.seed)
        for i, (train_df, val_df) in enumerate(splits, start=1):
            split_dir = os.path.join(args.dataset, f'split{i}')
            save_split(split_dir, train_df['name'], val_df['name'], test_df['name'])
            print(f"✓ Created split {i} with train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    elif args.split_method == "random":
        print(f"🎲 Performing StratifiedKFold-based splitting with {args.split} folds...")
        skf = StratifiedKFold(n_splits=args.split, shuffle=True, random_state=args.seed)
        for i, (train_idx, val_idx) in enumerate(skf.split(remaining_df['name'], remaining_df['bin']), start=1):
            train_df = remaining_df.iloc[train_idx]
            val_df = remaining_df.iloc[val_idx]
            split_dir = os.path.join(args.dataset, f'split{i}')
            save_split(split_dir, train_df['name'], val_df['name'], test_df['name'])
            print(f"✓ Created split {i} with train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    else:
        raise ValueError(f"Unknown split method: {args.split_method}")

def _filter_existing_test_names(test_names, existing_names_set):
    """Return (kept, dropped) after checking membership in images/ by base name."""
    test_names = [n.strip() for n in test_names if n.strip()]
    # de-dup while preserving order
    seen = set()
    test_names = [n for n in test_names if not (n in seen or seen.add(n))]

    kept = [n for n in test_names if n in existing_names_set]
    dropped = [n for n in test_names if n not in existing_names_set]
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val/test split based on label counts.")
    parser.add_argument('--dataset', default=config.output_directory,
                        help='Path to dataset folder containing images/ and csv/')
    parser.add_argument('--train_val_ratio', type=float, default=config.train_val_ratio,
                        help='Train/val split ratio (e.g., 0.85 means 85% train, 15% val)')
    parser.add_argument('--test_size', type=int, default=config.test_size,
                        help='Number of test images (ignored if test_fname is used)')
    parser.add_argument('--seed', type=int, default=config.seed,
                        help='Random seed for reproducibility')
    parser.add_argument('--split', type=int, default=config.split,
                        help='Number of splits')
    parser.add_argument('--config', type=str, default='config', help='Config module name (already processed)')
    parser.add_argument('--split_method', type=str, default=config.split_method, choices=['wasserstein', 'random'],
                    help='Splitting strategy: wasserstein or random (default: wasserstein)')

    args = parser.parse_args()
    run_split(args)

if __name__ == '__main__':
    main()
