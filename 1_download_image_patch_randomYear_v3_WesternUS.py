import ee
import geemap
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path # JS added
from geemap import download_ee_image  # required for new function

# Initialize file path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="naip-dead-trees")

# Set random seed for reproducibility (for selecting year)
np.random.seed(123) 

# User-defined parameters
# combination of state-years that have good data for labeling and inference.
state_years = {
    "AZ": [2015, 2019, 2021, 2023],
    "CA": [2009, 2012, 2014, 2016, 2018, 2020, 2022],
    "CO": [2011, 2015, 2017, 2019, 2021, 2023],
    "ID": [2011, 2015, 2017, 2019, 2021, 2023],
    "MT": [2009, 2013, 2015, 2017, 2019, 2021, 2023],
    "NM": [2011, 2014, 2016, 2018, 2020, 2022],
    "NV": [2015, 2017, 2019, 2022],
    "OR": [2009, 2012, 2014, 2016, 2020, 2022],
    "UT": [2011, 2014, 2016, 2018, 2021],
    "WA": [2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023],
    "WY": [2012, 2017, 2019, 2022]
}

scale = 1 #1 / 0.6 #For now, let's do 1m resolution because it is available across a larger number of years. In CO, it is available at 1m across 2009, 2011, 2013, 2015, and 2017. (so we will divide 20 images into each year)
img_dimension = [256, 256] #[256, 256] or [512, 512] #the CNN will input 256 regardless.
tg_crs = "EPSG:5070"

# Read CSV
input_path = PROJECT_ROOT / 'data' / 'western_us_stratified_sample_points_6-3-2026_seed_65.csv' 
df = pd.read_csv(input_path)
base_output_dir = PROJECT_ROOT / 'data' / 'images_to_label' / 'western_us_2009to2023_1m_strat_random'
os.makedirs(base_output_dir, exist_ok=True) 

# Get state abbreviation from TIGER dataset; outside loop so it doesn't have to get reconstructed each time we pull.
states = ee.FeatureCollection("TIGER/2018/States")

# Loop through dataframe
for idx, row in df.iterrows():
    if idx > 10000:
        break

    lat = row["lat"]
    lon = row["lon"]
    point = ee.Geometry.Point([lon, lat])

    state_feature = states.filterBounds(point).first()
    try:
        state_name = state_feature.get("STUSPS").getInfo()
    except:
        print(f"⚠️ Could not find state for point at index {idx}, skipping...")
        continue


    print(f"Index {idx} | Lat: {lat}, Lon: {lon} | State: {state_name}") #state_name is two digit state code.

    # Define region buffer
    point = ee.Geometry.Point([lon, lat])
    buffer_radius = np.floor((scale * img_dimension[1]) / 2)
    region = point.buffer(buffer_radius, proj=tg_crs).bounds(proj=tg_crs)

    # Get all available NAIP images for the location
    all_images = ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(point)

    # Extract distinct available years
    def extract_year(img):
        return ee.Feature(None, {'year': ee.Date(img.get('system:time_start')).get('year')})

    years = all_images.map(extract_year).aggregate_array('year').distinct().sort()
    
    try:
        # Access distinct available years for the imagery
        all_years = years.getInfo()
        
        # Get manually selected years for this state
        valid_years = state_years.get(state_name, [])

        # Keep only years that both:
        # 1. exist in NAIP for this point
        # 2. are approved for this state
        available_years = [y for y in all_years if y in valid_years]
        
    except Exception as e:
        print(f"Failed to get available years for index {idx}: {e}")
        continue

    if not available_years:
        print(f"No available NAIP years found for {state_name} at index {idx}." 
              f"Available years: {all_years}, Valid years: {valid_years}. Skipping...") #update to be from the appropriate year
        continue

    # Choose a random year from the filtered list
    year_str = str(np.random.choice(available_years))
    print(f"Selected random year: {year_str}")

    # Define output directory that includes state name and year
    output_dir = base_output_dir / state_name / f"{state_name}_{year_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output file path (after year is selected)
    out_file = os.path.join(output_dir, f"image_{idx}_{state_name}_{year_str}.tif")
    if os.path.exists(out_file):
        print(f"File already exists, skipping: {out_file}")
        continue

    # Filter image collection for that year ... here I could filter out later years...
    start_date = f"{year_str}-01-01"
    end_date = f"{year_str}-12-31"
    image_collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterDate(start_date, end_date)
        .filterBounds(region)
    )

    if image_collection.size().getInfo() == 0:
        print(f"No images found in {year_str} for index {idx}. Skipping...")
        continue

    # Take the first image (sorted by date) to extract its acquisition date
    first_img = image_collection.sort("system:time_start").first()
    actual_date_str = ee.Date(first_img.get("system:time_start")).format("YYYYMMdd").getInfo()

    # Define output file path with date included
    out_file = os.path.join(
        output_dir,
        f"image_{idx}_{state_name}_{year_str}_{actual_date_str}.tif"
    )
    if os.path.exists(out_file):
        print(f"File already exists, skipping: {out_file}")
        continue

    # Mosaic images for the whole year, but keep the date from the first image #do I want to be mosaicking across the whole year?
    proj = first_img.select(0).projection()
    image = image_collection.mosaic().setDefaultProjection(proj)

    # Export image using bilinear resampling
    try:
        download_ee_image(
            image=image,
            filename=out_file,
            region=region,
            scale=scale,
            crs=tg_crs,
            resampling="bilinear",
            overwrite=True
        )
        print(f"Exported with bilinear resampling: {out_file}\n")
    except Exception as e:
        print(f"Export failed for index {idx}: {e}")
