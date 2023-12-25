import os
import glob
import re
from caliop import Caliop_hdf_reader
from shapely.geometry import Point
import pandas as pd
import datetime
from pyhdf.HDF import *
from pyhdf.SD import *
from pyhdf.V import *
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
import tqdm
import argparse
    
def find_MODIS_files(folder_to_search, start_datetime, end_datetime, MODIS_product="MYD35"):
    search_pattern = "**/" + MODIS_product + "*.hdf"
    filepaths = glob.glob(search_pattern, root_dir=folder_to_search, recursive=True)

    filepaths_in_interval = []

    # make up a regex pattern
    pattern = re.compile(r'(?P<product_name>[^.]+?)\.A(?P<year>\d{4})(?P<day_of_year>\d{3})\.(?P<time>\d{4})\..+\.hdf')

    for file_index in range(len(filepaths)):
        filename = os.path.basename(filepaths[file_index])
        match = pattern.search(filename)

        if not match:
            # TODO set up a warning
            warning_flag = 1
            continue
        
        # Extract information from the matched groups
        product_name = match.group('product_name')
        year = int(match.group('year'))
        day_of_year = int(match.group('day_of_year'))
        hour = int(match.group('time')[:2])
        minute = int(match.group('time')[2:])

        # put everything together in a datetime object
        MODIS_frame_datetime = datetime.datetime(year=year, month=1, day=1, hour=hour, minute=minute) +\
                                datetime.timedelta(days=day_of_year-1)
        
        if (MODIS_frame_datetime + datetime.timedelta(minutes=5) < start_datetime) or (MODIS_frame_datetime > end_datetime):
            continue

        filepaths_in_interval.append(filepaths[file_index])
    
    return filepaths_in_interval

def collocate_pixels(caliop_long, caliop_lat, modis_long, modis_lat, k_neighbors=1):
    modis_long = modis_long.flatten()
    modis_lat = modis_lat.flatten()

    modis_kdtree = KDTree(list(zip(modis_long * np.cos(np.pi * modis_lat / 180), modis_lat)))
    caliop_query_points = list(zip(caliop_long * np.cos(np.pi * caliop_lat / 180), caliop_lat))

    # TODO check if caliop datapoints are enveloped by modis pixels
    _, indices = modis_kdtree.query(caliop_query_points, k=k_neighbors)

    return indices, modis_long[indices], modis_lat[indices]

def plot_collocation(CALIOP_filename, collocation_df):
    MODIS_filenames = collocation_df.MODIS_file.unique()
    number_of_found_MODIS_files = len(MODIS_filenames)

    caliop_central_long = (collocation_df.long.max() + collocation_df.long.min())/2
    caliop_central_lat = (collocation_df.lat.max() + collocation_df.lat.min())/2
    ccrs_projection = ccrs.Orthographic(central_longitude=caliop_central_long, central_latitude=caliop_central_lat)
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': ccrs_projection})
    ax.set_extent([collocation_df.long.min(), collocation_df.long.max(), collocation_df.lat.min(), collocation_df.lat.max()], ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=1)

    colors = ["darkblue", "orange"]

    for modis_filename, color in zip(MODIS_filenames, colors):
        ax.scatter(collocation_df.modis_long[collocation_df.MODIS_file == modis_filename],
                   collocation_df.modis_lat[collocation_df.MODIS_file == modis_filename],
                   label=modis_filename, transform=ccrs.PlateCarree(), c=color)
    
    ax.plot(collocation_df.long, collocation_df.lat, label="caliop track", c="r", transform=ccrs.PlateCarree())
    ax.set_title(CALIOP_filename)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle="dashed")
    ax.legend()

    if not os.path.exists("./collocation_images/"):
        os.mkdir("./collocation_images/")
    
    fig.savefig("./collocation_images/" + CALIOP_filename + ".png", dpi=90)
    plt.close(fig)

def collocate_CALIOP_with_MODIS_in_shape(CALIPSO_file, shape_polygon, csv_name,
                                         MODIS_pre_path="/neodc/modis/data/MYD03/collection61",
                                         delay_minutes=2.2,
                                         tolerance_minutes=0.5,
                                         save_img=False):
    
    reader = Caliop_hdf_reader()
    caliop_df = pd.DataFrame(columns=["long", "lat", "time", "profile_id"])
    caliop_df.long = reader._get_longitude(CALIPSO_file)
    caliop_df.lat = reader._get_latitude(CALIPSO_file)
    caliop_df.time = reader._get_profile_UTC(CALIPSO_file)
    caliop_df.profile_id = reader._get_profile_id(CALIPSO_file)
    caliop_df = caliop_df.set_index('profile_id')

    # prepare the coordinates in a GeoDataFrame to use the built in within() function
    track_coords_dict = {"geometry": [Point(long, lat) for long, lat in zip(caliop_df.long, caliop_df.lat)]}
    track_coords_gdf = gpd.GeoDataFrame(track_coords_dict, index=caliop_df.index)
    mask_over_Greenland = track_coords_gdf.within(shape_polygon)
    
    # if no datapoints over Greenland
    if not mask_over_Greenland.any():
        return "nothing over Greenland"

    start_datetime = caliop_df.time[mask_over_Greenland].iloc[0]\
        + datetime.timedelta(minutes = - delay_minutes - tolerance_minutes)
    end_datetime = caliop_df.time[mask_over_Greenland].iloc[-1]\
        + datetime.timedelta(minutes = - delay_minutes + tolerance_minutes)

    caliop_df = caliop_df.loc[mask_over_Greenland, :]

    # TODO deal with the case in which the start and end times are not within the same day
    # TODO this is specific to the CEDA archive on JASMIN, move to a function?
    MODIS_path = os.path.join(MODIS_pre_path, start_datetime.strftime("%Y"),\
                              start_datetime.strftime("%m"), start_datetime.strftime("%d"))
    
    found_MODIS_files = find_MODIS_files(MODIS_path, start_datetime, end_datetime, MODIS_product="MYD35")
    number_of_found_MODIS_files = len(found_MODIS_files)

    if number_of_found_MODIS_files == 0:
        return "no MODIS files found in time interval"

    if number_of_found_MODIS_files > 2:
        return "too many MODIS files found"

    if number_of_found_MODIS_files == 2:
        MODIS_reader1 = SD(os.path.join(MODIS_path, found_MODIS_files[0]))
        MODIS_reader2 = SD(os.path.join(MODIS_path, found_MODIS_files[1]))
        modis_long = np.concatenate([MODIS_reader1.select("Longitude").get(),\
                                    MODIS_reader2.select("Longitude").get()], axis=0)
        modis_lat = np.concatenate([MODIS_reader1.select("Latitude").get(),\
                                    MODIS_reader2.select("Latitude").get()], axis=0)
        
        number_of_pixels_in_one_file = int(np.size(modis_lat) / 2)

        # TODO deal with the case of missing MODIS files by calculating the average discrepancy between modis and caliop coords
        caliop_df["modis_idx"], caliop_df["modis_long"], caliop_df["modis_lat"] =\
            collocate_pixels(caliop_df.long, caliop_df.lat, modis_long, modis_lat)
        
        filename_list = pd.Series(found_MODIS_files[0], index=caliop_df.index)

        for id in caliop_df.index:
            if caliop_df.modis_idx[id] < number_of_pixels_in_one_file:
                file_no = 0
            else:
                file_no = 1
                caliop_df.loc[id, "modis_idx"] = caliop_df.loc[id, "modis_idx"] - number_of_pixels_in_one_file

            filename_list[id] = found_MODIS_files[file_no]

        caliop_df["MODIS_file"] = filename_list

    if number_of_found_MODIS_files == 1:
        MODIS_reader = SD(os.path.join(MODIS_path, found_MODIS_files[0]))
        modis_long = MODIS_reader.select("Longitude").get()
        modis_lat = MODIS_reader.select("Latitude").get()

        caliop_df["modis_idx"], caliop_df["modis_long"], caliop_df["modis_lat"] =\
            collocate_pixels(caliop_df.long, caliop_df.lat, modis_long, modis_lat)
        
        filename_list = pd.Series(found_MODIS_files[0], index=caliop_df.index)
        caliop_df["MODIS_file"] = filename_list

    collocation_path = os.path.join("./collocation_database",
                                    start_datetime.strftime("%Y"), start_datetime.strftime("%m"))

    if not os.path.exists(collocation_path):
        os.makedirs(collocation_path)

    caliop_df.to_csv(os.path.join(collocation_path, csv_name))

    if save_img:
        plot_collocation(csv_name[0:-4], caliop_df)

    return "ok"

def main(args):
    years = list(args.year)
    months = list(args.month)
    MODIS_folder = args.modisfolder

    # TODO multithreading implementation? each year/month on a thread?
    # TODO save logs to disk each iteration so a crash doesnt wipe it all out

    # JASMIN folder
    # CALIOP_folder = "/gws/nopw/j04/gbov/data/asdc.larc.nasa.gov/data/CALIPSO/LID_L2_05kmMLay-Standard-V4-51/"
    for year in years:
        for month in months:
            CALIOP_folder = os.path.join(args.caliopfolder, str(year), f"{month:02d}")

            if not os.path.exists(CALIOP_folder):
                print("Folder " + CALIOP_folder + " does not exist. Skipping month...")
                continue

            CALIOP_file_list = os.listdir(CALIOP_folder)
            collocation_outputs = [0] * len(CALIOP_file_list)
            print(f"There are {len(CALIOP_file_list)} files in {year}/{month:02d}.")

            # JASMIN folder
            # MODIS_folder = "/neodc/modis/data/MYD35_L2/collection61"

            greenland_geojson = gpd.read_file("Greenland_ALL.geojson")
            greenland_polygon = greenland_geojson.geometry[1]

            for caliop_file_count, CALIOP_file_path in tqdm.tqdm(enumerate(CALIOP_file_list)):
                collocation_outputs[caliop_file_count] = collocate_CALIOP_with_MODIS_in_shape(os.path.join(CALIOP_folder, CALIOP_file_path),
                                                            greenland_polygon,
                                                            CALIOP_file_path[0:-4] + ".csv",
                                                            MODIS_pre_path=MODIS_folder, save_img=True)
    
    collocation_logs = pd.DataFrame(collocation_outputs, index=CALIOP_file_list, columns=["output"])

    if not os.path.exists("./collocation_logs"):
        os.mkdir("./collocation_logs")
    collocation_logs.to_csv("./collocation_logs/collocation_log_" + datetime.datetime.now().strftime("%d_%m_%YT%H_%M_%S") +
                            ".csv")

def validate_year(value):
    ivalue = int(value)
    if 2006 <= ivalue <= 2023:
        return ivalue
    else:
        raise argparse.ArgumentTypeError(f"{value} is not a valid year. Must be between 2006 and 2023 (inclusive).")

def validate_month(value):
    ivalue = int(value)
    if 1 <= ivalue <= 12:
        return ivalue
    else:
        raise argparse.ArgumentTypeError(f"{value} is not a valid month. Must be between 1 and 12 (inclusive).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for collocating CALIOP with MODIS.")
    parser.add_argument("-year", type=validate_year, nargs="+", help="Years", default=[i for i in range(2006, 2024)])
    parser.add_argument("-month", type=validate_month, nargs="+", help="Months", default=[i for i in range(1, 13)])
    parser.add_argument("-caliopfolder", type=str, help="CALIOP folder", default="./test_data/CALIOP/")
    parser.add_argument("-modisfolder", type=str, help="MODIS folder", default="./test_data/MODIS/")

    args = parser.parse_args()
    main(args)