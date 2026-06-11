# Not optimal. Should do something to parallelize this script.

import os
import glob
import argparse
import rasterio
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from rasterio.features import rasterize
import pandas as pd
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import Point
from pyproj import Transformer
from tqdm import tqdm

from rasterio.enums import Resampling

import argparse

import sys, importlib
config_name = next((sys.argv[i + 1] for i, v in enumerate(sys.argv) if v == '--config' and i + 1 < len(sys.argv)), 'config')
config = importlib.import_module(config_name)
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

def reproject_to_resolution(src_path, resolution, resample_method, temp_dir="/tmp/reprojected"):
    """
    Reproject a raster to a given resolution.

    resample_method:
        - "bilinear": use bilinear for this image
        - "nearest": use nearest for this image
        - "mixed": randomly choose "bilinear" or "nearest" for this image
    """
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, os.path.basename(src_path))

    # Available methods
    resample_methods = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
    }

    # Determine the method to use for THIS image
    method_str = str(resample_method).lower()
    resample_choice = resample_methods[method_str]

    with rasterio.open(src_path) as src:
        # Compute new size
        scale_x = src.res[0] / float(resolution)
        scale_y = src.res[1] / float(resolution)
        new_height = max(1, int(round(src.height * scale_y)))
        new_width  = max(1, int(round(src.width  * scale_x)))

        # Read data with chosen resampling
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resample_choice
        )

        # Update transform
        new_transform = src.transform * Affine.scale(src.width / new_width, src.height / new_height)

        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform,
            "compress": "lzw",
        })

        with rasterio.open(temp_path, "w", **kwargs) as dst:
            dst.write(data)

    # print(f"Reprojected raster saved to {temp_path} using {method_str} resampling.")
    return temp_path




def match_crs(gdf, raster_path):
    with rasterio.open(raster_path) as src:
        # print(f"[DEBUG] Vector CRS: {gdf.crs}")
        # print(f"[DEBUG] Raster CRS: {src.crs}")
        return gdf.to_crs(src.crs)

def get_coord_id_from_bounds(bounds_tuple, src_crs):
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = bounds_tuple
    lon, lat = transformer.transform((minx + maxx) / 2, (miny + maxy) / 2)
    lat_str = f"{lat:.3f}".replace(".", "p").replace("-", "m")
    lon_str = f"{lon:.3f}".replace(".", "p").replace("-", "m")
    return f"{lat_str}_{lon_str}"

def process_image(tif_path, points_gdf, output_dir, generate_raster, res_str):
    filename = os.path.basename(tif_path)

    image_output = os.path.join(output_dir, "images")
    csv_output = os.path.join(output_dir, "csv")
    raster_output = os.path.join(output_dir, "labels") if generate_raster else None

    os.makedirs(image_output, exist_ok=True)
    os.makedirs(csv_output, exist_ok=True)
    if generate_raster:
        os.makedirs(raster_output, exist_ok=True)
    
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        bounds = src.bounds
        transform = src.transform
        shape = (src.height, src.width)
    
    base_name = filename.replace('.tif', '')
    if args.add_coord_id:
        coord_id = get_coord_id_from_bounds(bounds, src.crs)
        base_name = f"{base_name}_{coord_id}"

    out_tif_path = os.path.join(raster_output, f"{base_name}_{res_str}m.tif") if generate_raster else None
    image_out_path = os.path.join(image_output, f"{base_name}_{res_str}m.tif")
    out_csv_path = os.path.join(csv_output, f"{base_name}_{res_str}m.csv")
    if not os.path.exists(image_out_path):
        with rasterio.open(tif_path) as src:
            image_data = src.read()
            meta.update({"compress": "lzw"})
            with rasterio.open(image_out_path, "w", **meta) as dst:
                dst.write(image_data)
        print(f"🖼️ Saved full image: {image_out_path}")
    if generate_raster and os.path.exists(out_tif_path) and os.path.exists(out_csv_path):
        print(f"⏩ File already exists. Skipped (already exists): {filename}")
        return

    points_gdf = match_crs(points_gdf, tif_path)
    bbox = box(*bounds)
    subset = points_gdf[points_gdf.geometry.intersects(bbox)]

    pixel_coords = []
    if not subset.empty:
        for geom in subset.geometry:
            x_f, y_f = ~transform * (geom.x, geom.y)
            y, x = int(round(y_f)), int(round(x_f))
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                if image_data[y, x].sum() > 0:  # Check if pixel is not black
                    pixel_coords.append((x, y))  # Save as x, y
        

        if generate_raster:
            binary_mask = rasterize(
                ((geom, 1) for geom in subset.geometry),
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            meta.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
            with rasterio.open(out_tif_path, "w", **meta) as dst:
                dst.write(binary_mask, 1)
            print(f"🌳 Saved annotation: {out_tif_path}")

        if len(pixel_coords) > 0:
            df = pd.DataFrame(pixel_coords, columns=["x", "y"]).drop_duplicates()
            df.to_csv(out_csv_path, index=False)
            print(f"🌳 Saved point pixel coords: {out_csv_path}")
        else:
            print(f"⚠️ No tree points found, skipping CSV for: {filename}")

def tile_image_and_process(tif_path, points_gdf, output_dir, generate_raster, res_str, patch_size=256):
    filename = os.path.splitext(os.path.basename(tif_path))[0]

    image_output = os.path.join(output_dir, "images")
    csv_output = os.path.join(output_dir, "csv")
    raster_output = os.path.join(output_dir, "labels") if generate_raster else None

    os.makedirs(image_output, exist_ok=True)
    os.makedirs(csv_output, exist_ok=True)
    if generate_raster:
        os.makedirs(raster_output, exist_ok=True)

    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        height, width = src.height, src.width
        dtype = src.dtypes[0]
        full_transform = src.transform
        image = src.read()
        image = np.transpose(image, (1, 2, 0))

    pad_height = patch_size - height % patch_size if height % patch_size else 0
    pad_width = patch_size - width % patch_size if width % patch_size else 0
    padded_image = np.zeros((height + pad_height, width + pad_width, image.shape[2]), dtype=dtype)
    padded_image[:height, :width] = image

    points_gdf = match_crs(points_gdf, tif_path)

    num_rows = (height + pad_height) // patch_size
    num_cols = (width + pad_width) // patch_size

    for row in range(num_rows):
        for col in range(num_cols):
            y_off = row * patch_size
            x_off = col * patch_size

            window = Window(x_off, y_off, patch_size, patch_size)
            patch = padded_image[y_off:y_off+patch_size, x_off:x_off+patch_size, :]

            nonzero_fraction = np.count_nonzero(patch) / patch.size
            if nonzero_fraction < 0.1:
                continue

            patch_transform = full_transform * Affine.translation(x_off, y_off)
            patch_bounds = rasterio.windows.bounds(window, full_transform)
            
            patch_name = filename
            if args.add_coord_id:
                coord_id = get_coord_id_from_bounds(patch_bounds, meta['crs'])
                patch_name += f"_{coord_id}"
            if num_rows > 1 or num_cols > 1:
                patch_name += f"_r{row}_c{col}"

            out_tif_path = os.path.join(raster_output, f"{patch_name}_{res_str}m.tif") if generate_raster else None
            out_img_path = os.path.join(image_output, f"{patch_name}_{res_str}m.tif")
            out_csv_path = os.path.join(csv_output, f"{patch_name}_{res_str}m.csv")

            if os.path.exists(out_img_path) and os.path.exists(out_csv_path) and (not generate_raster or os.path.exists(out_tif_path)):
                # print(f"⏩ File already exists. Skipped tile: {patch_name}")
                continue

            meta.update({
                "height": patch_size,
                "width": patch_size,
                "transform": patch_transform,
                "count": patch.shape[2],
                "dtype": dtype,
                "compress": "lzw"
            })
            with rasterio.open(out_img_path, "w", **meta) as dst:
                dst.write(patch.transpose(2, 0, 1))
            # print(f"🌳 Saved image patch: {out_img_path}")

            bbox = box(*patch_bounds)
            subset = points_gdf[points_gdf.geometry.within(bbox)]
                            
            pixel_coords = []
            for geom in subset.geometry:
                x_f, y_f = ~patch_transform * (geom.x, geom.y)
                y_px, x_px = int(round(y_f)), int(round(x_f))
                if 0 <= y_px < patch_size and 0 <= x_px < patch_size:
                    # Check if pixel is not black (across all bands)
                    if patch[y_px, x_px].sum() > 0:
                        pixel_coords.append((x_px, y_px))


            if len(pixel_coords) > 0:
                df = pd.DataFrame(pixel_coords, columns=["x", "y"]).drop_duplicates()
                df.to_csv(out_csv_path, index=False)
                # print(f"🌳 Saved tree annotation csv: {out_csv_path}")
            # else:
                # print(f"⚠️ No tree points found, skipping CSV for: {patch_name}")

                
            ##### DEBUG
            # if len(subset) >= 0:
                # df = pd.DataFrame(pixel_coords, columns=["x", "y"]).drop_duplicates()
                diff = len(subset) - len(df)
                if diff != 0:
                    print(f"[DEBUG] {filename}: {len(subset)} (gt) - {len(df)} (csv) = {diff}")

                
                
            if generate_raster:
                if subset.empty:
                    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                else:
                    mask = rasterize(
                        ((geom, 1) for geom in subset.geometry),
                        out_shape=(patch_size, patch_size),
                        transform=patch_transform,
                        fill=0,
                        dtype=np.uint8
                    )
                meta.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": patch_transform,
                    "count": 1,
                    "dtype": "uint8",
                    "compress": "lzw"
                })
                with rasterio.open(out_tif_path, "w", **meta) as dst:
                    dst.write(mask, 1)
                # print(f"🌳 Saved tree annotation tif: {out_tif_path}")
                


def main(args):
    os.makedirs(args.output, exist_ok=True)
    labels_path = os.path.join(args.input, args.labels)
    images_path = os.path.join(args.input, args.images)

    points_gdf = gpd.read_file(labels_path, layer=args.layer)
    points_gdf = points_gdf.drop_duplicates(subset="geometry").reset_index(drop=True) # emove exact duplicate geometries

    tif_paths = glob.glob(os.path.join(images_path, "*.tif"))
    
    res_str = str(args.reproject_resolution).replace('.', 'd')
            
    for i, tif_path in enumerate(tqdm(tif_paths, desc="Progress", unit="image")):
        # Determine method for this image
        if args.resample_method.lower() == "mixed":
            method_for_this = "bilinear" if i % 2 == 0 else "nearest"
        else:
            method_for_this = args.resample_method.lower()

        # Log which method is used
        # print(f"Using -{method_for_this}- resampling for {os.path.basename(tif_path)}")

        # Reproject if requested
        if args.reproject_resolution:
            tif_path = reproject_to_resolution(tif_path, args.reproject_resolution, method_for_this)
        if args.tile:
            tile_image_and_process(tif_path, points_gdf, args.output, args.raster, res_str)
        else:
            process_image(tif_path, points_gdf, args.output, args.raster, res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert tree points to binary masks and/or pixel CSVs.")
    parser.add_argument("--input", default=config.input_directory, help="Input directory path of original images and labels")
    parser.add_argument("--images", default=config.image_folder_name, help="Folder name of input images")
    parser.add_argument("--labels", default=config.label_file_name, help="Folder name of input labels")
    parser.add_argument("--output", default=config.output_directory, help="Output directory path of organized images and labels")
    parser.add_argument("--reproject_resolution", type=float, default=config.reproject_resolution,
                    help="Target resolution in meters for optional reprojection")
    parser.add_argument("--resample_method", type=str, default=config.resample_method,
                help="Resampling method for optional reprojection")
    parser.add_argument("--layer", default=None, help="Layer name in GPKG (default is first layer)")
    parser.add_argument("--raster", action="store_true", help="If set, generate .tif annotation mask")
    parser.add_argument("--tile", action="store_true", help="If set, tile images into 256x256 patches with padding")
    parser.add_argument("--add_coord_id", action="store_true", help="If set, include lat/lon ID in output filenames")
    parser.add_argument('--config', type=str, default='config', help='Config module name (already processed)')

    parser.set_defaults(tile=config.tile)

    args = parser.parse_args()
    main(args)