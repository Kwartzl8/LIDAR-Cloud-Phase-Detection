from pyhdf.HDF import *
from pyhdf.SD import *
from pyhdf.V import *
import os
import pandas as pd
import cartopy.crs as ccrs
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import glob

# path_to_2017_collocation_database = \
#     "/Users/alexandrudobra/University/MPhys/Project/MPhys_repo/LIDAR-Cloud-Phase-Detection/collocation_database/2017/01"
# path_to_2017_collocation_database = \
#     "/Users/alexandrudobra/University/MPhys/Project/collocation_testing/JASMIN_tests/hospital01/collocation_testing/collocation_database"
# path_to_MODIS_folder = "./test_data/MODIS/2017/01/01/"

path_to_2017_collocation_database = \
    "./collocation_database/"

path_to_MODIS_folder = "/neodc/modis/data/MYD35_L2/collection61/2017/01/01"

def get_sensor_zenith_angle(folder, collocation):
    modis_files = collocation.MODIS_file.unique()
    readers = {}
    search_strings = {}
    pixel_angles = np.full(len(collocation.index), -1)

    for modis_file in modis_files:
        search_strings[modis_file] = "*" + modis_file[9:22] + "*"

        myd06_file = glob.glob(search_strings[modis_file], root_dir=folder)
        
        if myd06_file == []:
            continue

        readers[modis_file] = SD(os.path.join(folder, myd06_file[0]))
        view_zenith_angle = np.array(readers[modis_file].select("Sensor_Zenith").get()).flatten()
        pixel_angles[collocation.MODIS_file == modis_file] = view_zenith_angle[collocation[collocation.MODIS_file == modis_file].modis_idx]

    return pixel_angles


collocation_csv_filelist = glob.glob("*.csv", root_dir=path_to_2017_collocation_database)

total_collocated_pixels = 0

ccrs_projection = ccrs.Orthographic(central_longitude=-40, central_latitude=75)
fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': ccrs_projection})
ax.coastlines(resolution='50m', color='black', linewidth=1)
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="dashed")

longs, lats = np.empty((0,)), np.empty((0,))
days = np.empty((0,))
pixel_angles = np.zeros((0,))

for file_index, file in tqdm.tqdm(enumerate(collocation_csv_filelist)):
    collocation = pd.read_csv(os.path.join(path_to_2017_collocation_database, file))
    total_collocated_pixels += len(collocation.index)

    pixel_angles = np.concatenate([pixel_angles, (get_sensor_zenith_angle(path_to_MODIS_folder, collocation))])

    days = np.concatenate((days,[date.day for date in pd.to_datetime(collocation.time)]))
    longs = np.concatenate((longs, collocation.long))
    lats = np.concatenate((lats, collocation.lat))

pos = ax.scatter(longs, lats, c=days, s=2, transform=ccrs.PlateCarree(), cmap="autumn")
fig.colorbar(pos, label="day", ticks=range(1, 32, 5))

print(total_collocated_pixels)

pixel_angles = pixel_angles[pixel_angles != -1] / 100
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(pixel_angles)
ax.set_title("View Zenith Angle Histogram, January 2017")
ax.set_xlabel("Angle (degrees)")
ax.set_ylabel("Count")
fig.savefig("view_zenith_angle_histogram_2017_01.png", dpi=200)