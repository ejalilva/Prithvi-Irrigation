#!/usr/bin/env python
# coding: utf-8

# In[3]:


# reading crop classes chips
import os
import glob
import rioxarray as rio
import rasterio
from rasterio.merge import merge
import xarray as xr
import numpy as np



file_id = 50

train_dir = '/discover/nobackup/ejalilva/data/prithvi/datasets--ibm-nasa-geospatial--multi-temporal-crop-classification/snapshots/04b439f179e52a7b144f69676210eecd30c39cfc/training_chips/'
val_dir = '/discover/nobackup/ejalilva/data/prithvi/datasets--ibm-nasa-geospatial--multi-temporal-crop-classification/snapshots/04b439f179e52a7b144f69676210eecd30c39cfc/validation_chips/'
cdl20_path = '/discover/nobackup/ejalilva/data/irrigation_FM/2020_30m_cdls.tif'
lgrip_tiles_dir = '/discover/nobackup/ejalilva/data/irrigation_FM'
files = glob.glob(os.path.join(train_dir,'*mask.tif'))
print(len(files))


# In[4]:


original_chip = rio.open_rasterio(files[file_id])
chip_crs = str(original_chip.rio.crs)
chip = original_chip.sel(band=1).drop_vars('band')
chip.x


# In[ ]:


# Cell 1: Irrigation likelihood mapping based on crop type
import rasterio
import numpy as np
import glob

# Irrigation likelihood classes:
# 0: No Data/Non-Ag
# 1: Highly Likely Irrigated 
# 2: Likely Irrigated (regional dependent)
# 3: Possibly Irrigated 
# 4: Unlikely Irrigated
# 5: Non-Agricultural

# Crop-based irrigation likelihood mapping
irrigation_likelihood_map = {
    # No Data / Background
    0: 0,   # No Data
    81: 0,  # Clouds/No Data
    
    # HIGHLY LIKELY IRRIGATED (almost always irrigated)
    3: 1,   # Rice
    47: 1,  # Misc Vegs & Fruits
    48: 1,  # Watermelons
    49: 1,  # Onions
    50: 1,  # Cucumbers
    54: 1,  # Tomatoes
    66: 1,  # Cherries
    67: 1,  # Peaches
    68: 1,  # Apples
    69: 1,  # Grapes
    70: 1,  # Christmas Trees
    71: 1,  # Other Tree Crops
    72: 1,  # Citrus
    74: 1,  # Pecans
    75: 1,  # Almonds
    76: 1,  # Walnuts
    77: 1,  # Pears
    92: 1,  # Aquaculture
    204: 1, # Pistachios
    206: 1, # Carrots
    207: 1, # Asparagus
    208: 1, # Garlic
    209: 1, # Cantaloupes
    210: 1, # Prunes
    211: 1, # Olives
    212: 1, # Oranges
    213: 1, # Honeydew Melons
    214: 1, # Broccoli
    215: 1, # Avocados
    216: 1, # Peppers
    217: 1, # Pomegranates
    218: 1, # Nectarines
    219: 1, # Greens
    220: 1, # Plums
    221: 1, # Strawberries
    222: 1, # Squash
    223: 1, # Apricots
    227: 1, # Lettuce
    229: 1, # Pumpkins
    242: 1, # Blueberries
    243: 1, # Cabbage
    244: 1, # Cauliflower
    245: 1, # Celery
    246: 1, # Radishes
    247: 1, # Turnips
    248: 1, # Eggplants
    249: 1, # Gourds
    250: 1, # Cranberries
    
    # LIKELY IRRIGATED (regional/high value crops)
    1: 2,   # Corn (regional dependent)
    2: 2,   # Cotton (varies by region)
    41: 2,  # Sugarbeets
    43: 2,  # Potatoes
    45: 2,  # Sugarcane
    46: 2,  # Sweet Potatoes
    36: 2,  # Alfalfa (often irrigated)
    12: 2,  # Sweet Corn
    
    # POSSIBLY IRRIGATED (supplemental irrigation)
    4: 3,   # Sorghum
    10: 3,  # Peanuts
    42: 3,  # Dry Beans
    37: 3,  # Other Hay/Non Alfalfa
    62: 3,  # Pasture/Grass
    
    # UNLIKELY IRRIGATED (typically rainfed)
    5: 4,   # Soybeans
    21: 4,  # Barley
    22: 4,  # Durum Wheat
    23: 4,  # Spring Wheat
    24: 4,  # Winter Wheat
    25: 4,  # Other Small Grains
    27: 4,  # Rye
    28: 4,  # Oats
    31: 4,  # Canola
    32: 4,  # Flaxseed
    44: 4,  # Other Crops
    61: 4,  # Fallow/Idle Cropland
    58: 4,  # Clover/Wildflowers
    59: 4,  # Sod/Grass Seed
    176: 4, # Grassland/Pasture
    
    # NON-AGRICULTURAL
    63: 5,  # Forest
    64: 5,  # Shrubland
    65: 5,  # Barren
    82: 5,  # Developed
    83: 5,  # Water
    87: 5,  # Wetlands
    111: 5, # Open Water
    121: 5, # Developed/Open Space
    122: 5, # Developed/Low Intensity
    123: 5, # Developed/Med Intensity
    124: 5, # Developed/High Intensity
    131: 5, # Barren
    141: 5, # Deciduous Forest
    142: 5, # Evergreen Forest
    143: 5, # Mixed Forest
    152: 5, # Shrubland
    190: 5, # Woody Wetlands
    195: 5, # Herbaceous Wetlands
}


# In[15]:


# Cell 2: Process chips with location-based adjustments
chip_files = glob.glob(f"{val_dir}/*mask.tif")
cdl_src = rasterio.open(cdl20_path)

print(f"Processing {len(chip_files)} chips...")

for i, chip_file in enumerate(chip_files):
    with rasterio.open(chip_file) as chip:
        # Get chip location (center coordinates) - keep in CDL coordinate system
        bounds = chip.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        
        # Transform regional boundaries to CDL coordinate system
        from rasterio.warp import transform_bounds
        
        # Western US boundary (longitude < -100) to CDL coordinates
        western_bounds = transform_bounds('EPSG:4326', cdl_src.crs, -180, 25, -100, 50)
        western_x_max = western_bounds[2]

        # High Plains boundary (-104 to -96 lon, 36 to 42 lat) to CDL coordinates  
        plains_bounds = transform_bounds('EPSG:4326', cdl_src.crs, -104, 36, -96, 42)
        plains_x_min, plains_y_min, plains_x_max, plains_y_max = plains_bounds
        
        # California Central Valley (-122 to -119 lon, 35 to 40 lat) to CDL coordinates
        ca_bounds = transform_bounds('EPSG:4326', cdl_src.crs, -122, 35, -119, 40)
        ca_x_min, ca_y_min, ca_x_max, ca_y_max = ca_bounds
        
        # Midwest Corn Belt (-98 to -80 lon, 38 to 46 lat) to CDL coordinates
        midwest_bounds = transform_bounds('EPSG:4326', cdl_src.crs, -98, 38, -80, 46)
        midwest_x_min, midwest_y_min, midwest_x_max, midwest_y_max = midwest_bounds
        
        # Extract from CDL at same location
        window = rasterio.windows.from_bounds(*chip.bounds, cdl_src.transform)
        cdl_data = cdl_src.read(1, window=window)
        
        # Create irrigation likelihood array
        irrigation_likelihood = np.full_like(cdl_data, 3, dtype=np.uint8)  # Default: possibly irrigated
        
        # Apply crop-based classification
        for cdl_code, likelihood in irrigation_likelihood_map.items():
            mask = cdl_data == cdl_code
            irrigation_likelihood[mask] = likelihood
        
        # Location-based adjustments using CDL coordinate system
        # Western US (more irrigation)
        if center_x < western_x_max:  # West of 100th meridian in CDL coordinates
            # Upgrade corn and cotton to highly likely irrigated in western US
            corn_mask = (cdl_data == 1) | (cdl_data == 12)  # Corn types
            cotton_mask = cdl_data == 2  # Cotton
            irrigation_likelihood[corn_mask] = 1  # Highly likely
            irrigation_likelihood[cotton_mask] = 1  # Highly likely
            
            # Upgrade other crops that are regionally likely
            regional_crops = [4, 24, 5]  # Sorghum, winter wheat, soybeans
            for crop_code in regional_crops:
                crop_mask = cdl_data == crop_code
                current_class = irrigation_likelihood[crop_mask]
                # Upgrade by one class (but don't exceed highly likely)
                irrigation_likelihood[crop_mask] = np.maximum(current_class - 1, 1)
        
        # High Plains (heavy irrigation region)
        if (plains_x_min < center_x < plains_x_max and 
            plains_y_min < center_y < plains_y_max):  # High Plains in CDL coordinates
            # Most crops become likely or highly likely irrigated
            crop_mask = (cdl_data >= 1) & (cdl_data <= 60)  # All crop codes
            current_class = irrigation_likelihood[crop_mask]
            irrigation_likelihood[crop_mask] = np.maximum(current_class - 1, 1)
        
        # California Central Valley (almost everything irrigated)
        if (ca_x_min < center_x < ca_x_max and 
            ca_y_min < center_y < ca_y_max):  # CA Central Valley in CDL coordinates
            crop_mask = (cdl_data >= 1) & (cdl_data <= 254)  # All crops
            irrigation_likelihood[crop_mask] = 1  # Highly likely
        
        # Midwest corn belt adjustments
        if (midwest_x_min < center_x < midwest_x_max and 
            midwest_y_min < center_y < midwest_y_max):  # Midwest in CDL coordinates
            # Soybeans and corn more likely rainfed in good rainfall areas
            soy_mask = cdl_data == 5  # Soybeans
            irrigation_likelihood[soy_mask] = 4  # Unlikely irrigated
            
            corn_mask = (cdl_data == 1) | (cdl_data == 12)  # Corn
            irrigation_likelihood[corn_mask] = 3  # Possibly irrigated
        
        # Save original CDL chip (with original classes)
        cdl_output_path = f"{lgrip_tiles_dir}/val_chip_20/cdl_original_{i:04d}.tif"
        with rasterio.open(cdl_output_path, 'w', driver='GTiff', 
                          height=cdl_data.shape[0], width=cdl_data.shape[1],
                          count=1, dtype=cdl_data.dtype, crs=cdl_src.crs,
                          transform=rasterio.windows.transform(window, cdl_src.transform)) as dst:
            dst.write(cdl_data, 1)
        
        # Save irrigation likelihood map
        irrigation_output_path = f"{lgrip_tiles_dir}/val_chip_20/irrigation_likelihood_{i:04d}.tif"
        with rasterio.open(irrigation_output_path, 'w', driver='GTiff', 
                          height=irrigation_likelihood.shape[0], width=irrigation_likelihood.shape[1],
                          count=1, dtype=irrigation_likelihood.dtype, crs=cdl_src.crs,
                          transform=rasterio.windows.transform(window, cdl_src.transform)) as dst:
            dst.write(irrigation_likelihood, 1)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(chip_files)}")

print("Done! Irrigation likelihood maps saved.")
print("Classes: 0=No Data, 1=Highly Likely, 2=Likely, 3=Possibly, 4=Unlikely, 5=Non-Ag")
cdl_src.close()


# In[16]:


chip_file


# In[17]:


import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rio
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Load data
cdl_output_path = f"{lgrip_tiles_dir}/val_chip_20/cdl_original_0770.tif"
irrigation_output_path = f"{lgrip_tiles_dir}/val_chip_20/irrigation_likelihood_0770.tif"

original_chip = rio.open_rasterio(cdl_output_path)
irrigation_likelihood = rio.open_rasterio(irrigation_output_path)

# Irrigation colormap
irrigation_colors = ['#2c3e50', '#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#95a5a6']
irrigation_labels = ["No Data", "Highly Likely", "Likely", "Possibly", "Unlikely", "Non-Ag"]
irrigation_cmap = ListedColormap(irrigation_colors)

# Get all unique CDL classes
cdl_data = original_chip.values
unique_classes = np.unique(cdl_data[~np.isnan(cdl_data)]).astype(int)

# CDL class names (add more as needed)
cdl_names = {
    1: 'Corn', 5: 'Soybeans', 12: 'Sweet Corn', 24: 'Winter Wheat', 27: 'Rye', 
    28: 'Oats', 36: 'Alfalfa', 37: 'Other Hay', 42: 'Dry Beans', 43: 'Potatoes',
    44: 'Other Crops', 59: 'Sod/Grass', 61: 'Fallow', 76: 'Walnuts', 
    111: 'Open Water', 121: 'Developed/Open', 122: 'Developed/Low', 
    123: 'Developed/Med', 124: 'Developed/High', 131: 'Barren', 
    141: 'Deciduous Forest', 142: 'Evergreen Forest', 143: 'Mixed Forest',
    152: 'Shrubland', 176: 'Grassland', 190: 'Woody Wetlands', 195: 'Herbaceous Wetlands'
}

# Create diverse contrasting colormap for CDL classes
# Combine multiple qualitative colormaps for maximum contrast
import matplotlib.cm as cm

n_classes = len(unique_classes)
print(f"Creating colormap for {n_classes} classes")

# Get colors from multiple qualitative colormaps for maximum diversity
colors1 = cm.tab20(np.linspace(0, 1, 20))  # 20 colors
colors2 = cm.Set1(np.linspace(0, 1, 9))    # 9 colors  
colors3 = cm.Set2(np.linspace(0, 1, 8))    # 8 colors
colors4 = cm.Set3(np.linspace(0, 1, 12))   # 12 colors
colors5 = cm.Paired(np.linspace(0, 1, 12)) # 12 colors

# Combine all colors and take what we need
all_colors = np.vstack([colors1, colors2, colors3, colors4, colors5])
selected_colors = all_colors[:n_classes]

# Create custom discrete colormap
cdl_cmap = ListedColormap(selected_colors)

# Create figure
fig, axes = plt.subplots(ncols=2, figsize=(18, 7))

# Left: Irrigation likelihood
im1 = irrigation_likelihood.plot(x='x', cmap=irrigation_cmap, add_colorbar=False, 
                                ax=axes[0], vmin=-0.5, vmax=5.5)
cbar1 = plt.colorbar(im1, ax=axes[0], ticks=np.arange(6))
cbar1.ax.set_yticklabels(irrigation_labels)
axes[0].set_title('Irrigation Likelihood')

# Right: CDL with diverse contrasting colors
im2 = original_chip.plot(x='x', cmap=cdl_cmap, add_colorbar=False, ax=axes[1],
                        vmin=unique_classes.min()-0.5, vmax=unique_classes.max()+0.5)

# Create legend for ALL unique classes with the diverse colors
legend_patches = []
for i, code in enumerate(sorted(unique_classes)):
    color = selected_colors[i]  # Use the exact color from our custom colormap
    name = cdl_names.get(code, f'Class {code}')
    patch = mpatches.Patch(color=color, label=f'{code}: {name}')
    legend_patches.append(patch)

axes[1].legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.05, 0.5), 
              fontsize=8, title=f'All CDL Classes ({len(unique_classes)} total)')
axes[1].set_title(f'CDL Classification - All {len(unique_classes)} Classes')

plt.tight_layout()
plt.show()

print(f"CDL Classes: {len(unique_classes)} unique classes ({unique_classes.min()}-{unique_classes.max()})")
print(f"Irrigation: {len(np.unique(irrigation_likelihood.values))} classes")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rio
from matplotlib.colors import ListedColormap

# Load data
original_chip = rio.open_rasterio(chip_file)  # Original CDL
irrigation_likelihood = rio.open_rasterio(output_path)  # Irrigation likelihood

# Colors for irrigation likelihood (6 classes: 0-5)
irrigation_colors = [
    '#2c3e50',  # 0: No Data
    '#e74c3c',  # 1: Highly Likely Irrigated
    '#f39c12',  # 2: Likely Irrigated  
    '#f1c40f',  # 3: Possibly Irrigated
    '#2ecc71',  # 4: Unlikely Irrigated
    '#95a5a6'   # 5: Non-Agricultural
]

irrigation_labels = [
    "No Data", "Highly Likely", "Likely", "Possibly", "Unlikely", "Non-Ag"
]

irrigation_cmap = ListedColormap(irrigation_colors)

# Get CDL range
original_data = original_chip.values[~np.isnan(original_chip.values)]
cdl_min, cdl_max = int(original_data.min()), int(original_data.max())

# Create plots
fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

# Left: Irrigation likelihood with labels
im1 = irrigation_likelihood.plot(x='x', cmap=irrigation_cmap, add_colorbar=False, 
                                ax=axes[0], vmin=-0.5, vmax=5.5)
cbar1 = plt.colorbar(im1, ax=axes[0], ticks=np.arange(6))
cbar1.ax.set_yticklabels(irrigation_labels)
axes[0].set_title('Irrigation Likelihood')

# Right: CDL continuous without labels
original_chip.plot(x='x', cmap='viridis', ax=axes[1], vmin=cdl_min, vmax=cdl_max,
                   cbar_kwargs={'label': 'CDL Code'})
axes[1].set_title('CDL Classification')

plt.tight_layout()
plt.show()


# In[ ]:


# Reclassification mapping: CDL original codes -> simplified classes
reclassify_map = {
    # No Data / Background
    0: 0,   # No Data
    81: 0,  # Clouds/No Data
    
    # Natural Vegetation  
    64: 1,  # Shrubland
    152: 1, # Shrubland
    176: 1, # Grassland/Pasture
    62: 1,  # Pasture/Grass
    
    # Forest
    63: 2,  # Forest
    141: 2, # Deciduous Forest
    142: 2, # Evergreen Forest
    143: 2, # Mixed Forest
    
    # Corn
    1: 3,   # Corn
    12: 3,  # Sweet Corn
    13: 3,  # Pop or Orn Corn
    
    # Soybeans
    5: 4,   # Soybeans
    
    # Wetlands
    87: 5,  # Wetlands
    190: 5, # Woody Wetlands
    195: 5, # Herbaceous Wetlands
    
    # Developed/Barren
    82: 6,  # Developed
    121: 6, # Developed/Open Space
    122: 6, # Developed/Low Intensity
    123: 6, # Developed/Med Intensity
    124: 6, # Developed/High Intensity
    65: 6,  # Barren
    131: 6, # Barren
    
    # Open Water
    83: 7,  # Water
    111: 7, # Open Water
    
    # Winter Wheat
    24: 8,  # Winter Wheat
    
    # Alfalfa
    36: 9,  # Alfalfa
    
    # Fallow/Idle Cropland
    61: 10, # Fallow/Idle Cropland
    
    # Cotton
    2: 11,  # Cotton
    
    # Sorghum
    4: 12,  # Sorghum
}

# Cell 2: Function to reclassify array
def reclassify_cdl(data_array, reclassify_dict):
    """Reclassify CDL data using mapping dictionary"""
    # Create output array, default to class 13 (Other)
    output = np.full_like(data_array, 13, dtype=np.uint8)
    
    # Apply reclassification
    for original_code, new_code in reclassify_dict.items():
        mask = data_array == original_code
        output[mask] = new_code
    
    return output

# Cell 3: Extract and reclassify CDL chips
chip_files = glob.glob(f"{val_dir}/*mask.tif")
cdl_src = rasterio.open(cdl20_path)

print(f"Processing {len(chip_files)} chips...")

for i, chip_file in enumerate(chip_files):
    with rasterio.open(chip_file) as chip:
        # Extract from CDL at same location
        window = rasterio.windows.from_bounds(*chip.bounds, cdl_src.transform)
        cdl_data = cdl_src.read(1, window=window)
        
        # Reclassify to match your 14-class scheme
        reclassified_data = reclassify_cdl(cdl_data, reclassify_map)
        
        # Save reclassified chip
        output_path = f"{lgrip_tiles_dir}/val_chip_20/cdl_2020_chip_{i:04d}.tif"
        with rasterio.open(output_path, 'w', driver='GTiff', 
                          height=reclassified_data.shape[0], width=reclassified_data.shape[1],
                          count=1, dtype=reclassified_data.dtype, crs=cdl_src.crs,
                          transform=rasterio.windows.transform(window, cdl_src.transform)) as dst:
            dst.write(reclassified_data, 1)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(chip_files)}")

print("Done! Reclassified CDL 2020 chips saved.")
cdl_src.close()


# In[ ]:


cdl_props = []
cdl_files = glob.glob(f"{lgrip_tiles_dir}/training_chip_20/*.tif")
len(cdl_files)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from matplotlib.colors import ListedColormap

# Load data (make sure both are loaded)
original_chip = rio.open_rasterio(chip_file)
new_chip = rio.open_rasterio(output_path)
colors = [
    '#dcdcdc',  # No Data - light grey
    '#76c476',  # Natural Vegetation - light green
    '#005700',  # Forest - dark green
    '#ffff00',  # Corn - yellow
    '#ffffb2',  # Soybeans - pale yellow
    '#7fffd4',  # Wetlands - aquamarine
    '#d3d3d3',  # Developed / Barren - grey
    '#1f77b4',  # Open Water - blue
    '#d2b48c',  # Winter Wheat - light brown
    '#347235',  # Alfalfa - rich green
    '#e4d96f',  # Fallow / Idle Cropland - tan
    '#f5f5f5',  # Cotton - light grey
    '#e4c400',  # Sorghum - deep yellow
    '#800080'   # Other - purple
]
cmap = ListedColormap(colors)

fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

# Plot new_chip
im1 = new_chip.plot(
    x='x',
    cmap=cmap,
    add_colorbar=False,
    ax=axes[0],
    vmin=-0.5,
    vmax=13.5
)
cbar1 = plt.colorbar(im1, ax=axes[0], ticks=np.arange(0, 14))
cbar1.ax.set_yticklabels([
    "No data", "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed / Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow / Idle Cropland", "Cotton", "Sorghum", "Other"
])
axes[0].set_title('New Chip CDL Crop Class')
axes[0].set_xlabel('X Coordinate')
axes[0].set_ylabel('Y Coordinate')

# Plot original_chip
im2 = original_chip.plot(
    x='x',
    cmap=cmap,
    add_colorbar=False,
    ax=axes[1],
    vmin=-0.5,
    vmax=13.5
)
cbar2 = plt.colorbar(im2, ax=axes[1], ticks=np.arange(0, 14))
cbar2.ax.set_yticklabels([
    "No data", "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed / Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow / Idle Cropland", "Cotton", "Sorghum", "Other"
])
axes[1].set_title('Original Chip CDL Crop Class')
axes[1].set_xlabel('X Coordinate')
axes[1].set_ylabel('Y Coordinate')

plt.tight_layout()
fig.autofmt_xdate()
plt.show()


# In[ ]:




