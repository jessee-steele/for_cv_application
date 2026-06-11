import ee
import geemap
import os
from pathlib import Path
from geemap import download_ee_image

# --------------------------------
# Initialize Earth Engine
# --------------------------------
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize(project="naip-dead-trees")

# --------------------------------
# Output directory (LOCAL)
# --------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
output_dir = PROJECT_ROOT / "data" / "NAIP_2021_CO_allstate"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------
# Colorado boundary
# --------------------------------
states = ee.FeatureCollection("TIGER/2018/States")
colorado = states.filter(ee.Filter.eq("STUSPS", "CO"))

# --------------------------------
# NAIP 2021 (NO MOSAIC)
# --------------------------------
naip_2021 = (
    ee.ImageCollection("USDA/NAIP/DOQQ")
    .filterBounds(colorado)
    .filter(ee.Filter.calendarRange(2021, 2021, "year"))
)

img_list = naip_2021.toList(naip_2021.size())
count = naip_2021.size().getInfo()

print(f"Found {count} NAIP tiles for Colorado (2021)")

# --------------------------------
# Helper: split geometry into grid
# --------------------------------
def split_geometry(geom, nx=3, ny=3):
    # Get bounding box coordinates (server-side)
    coords = ee.List(geom.bounds().coordinates().get(0))

    ll = ee.List(coords.get(0))  # lower-left
    ur = ee.List(coords.get(2))  # upper-right

    xmin = ee.Number(ll.get(0))
    ymin = ee.Number(ll.get(1))
    xmax = ee.Number(ur.get(0))
    ymax = ee.Number(ur.get(1))

    dx = xmax.subtract(xmin).divide(nx)
    dy = ymax.subtract(ymin).divide(ny)

    cells = []
    for i in range(nx):
        for j in range(ny):
            cell = ee.Geometry.Rectangle([
                xmin.add(dx.multiply(i)),
                ymin.add(dy.multiply(j)),
                xmin.add(dx.multiply(i + 1)),
                ymin.add(dy.multiply(j + 1))
            ])
            cells.append(cell)

    return cells

# --------------------------------
# Export each NAIP tile as 3x3 subtiles
# --------------------------------
for i in range(count):
    img = ee.Image(img_list.get(i))

    img_id = img.get("system:index").getInfo()
    #date_str = ee.Date(img.get("system:time_start")).format("YYYYMMdd").getInfo()

    print(f"\nProcessing {i+1}/{count}: {img_id}")

    tiles = split_geometry(img.geometry(), nx=3, ny=3)

    for t, tile_geom in enumerate(tiles):
        out_file = output_dir / f"NAIP_2021_CO_{img_id}_tile{t}.tif"

        if out_file.exists():
            print(f"  Skipping existing tile: {out_file.name}")
            continue

        print(f"  Downloading tile {t+1}/9")

        tile_geom = tile_geom.intersection(img.geometry(), 1)
        download_ee_image(
            image=img,
            filename=out_file,
            region=tile_geom,
            scale=0.6,                # native NAIP resolution
            crs="EPSG:5070",
            resampling="bilinear",
            overwrite=True
        )