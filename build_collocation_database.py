import os
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

def find_MODIS_files_in_interval(folder_to_search, start_datetime, end_datetime,
                        delay_minutes,
                        tolerance_minutes):

    # if tolerance_minutes > 2.5, we can get more than 2 modis files, which breaks things
    # TODO implement error message
    
    # consider delay between satellites and give some tolerance
    lower_bound = start_datetime + datetime.timedelta(minutes = - delay_minutes - tolerance_minutes)
    upper_bound = end_datetime + datetime.timedelta(minutes = - delay_minutes + tolerance_minutes)
    
    # round down to 5 mins, as that is how modis data is chunked
    from math import floor
    lower_bound = lower_bound.replace(minute=floor(lower_bound.minute / 5) * 5, second=0, microsecond=0)
    upper_bound = upper_bound.replace(minute=floor(upper_bound.minute / 5) * 5, second=0, microsecond=0)

    modis_files_selected = int(((upper_bound - lower_bound).seconds / 60) / 5 + 1)

    search_strings = [""] * modis_files_selected
    filename_list = [""] * modis_files_selected
    
    import glob
    for i in range(modis_files_selected):
        # construct the search strings in the format \AYYYYDDD\.HHMM where DDD is the day of the year
        modis_start_time = lower_bound + datetime.timedelta(minutes=5*i)
        day_of_year = (modis_start_time - modis_start_time.replace(month=1, day=1)).days + 1
        search_strings[i] = "A" + str(modis_start_time.year) + str(day_of_year).zfill(3) + "." +\
                            str(modis_start_time.hour).zfill(2) + str(modis_start_time.minute).zfill(2)
        
        found_files = glob.glob("**/*" + search_strings[i] + "*.hdf", root_dir=folder_to_search, recursive=True)

        if len(found_files) != 1:
            # something went terribly wrong
            # TODO have the case where no files are found or when multiples files are found
            continue
        
        filename_list[i] = found_files[0]
    
    return search_strings, filename_list

def collocate_pixels(caliop_long, caliop_lat, modis_long, modis_lat, k_neighbors=1):
    modis_long = modis_long.flatten()
    modis_lat = modis_lat.flatten()

    modis_kdtree = KDTree(list(zip(modis_long * np.cos(np.pi * modis_lat / 180), modis_lat)))
    caliop_query_points = list(zip(caliop_long * np.cos(np.pi * caliop_lat / 180), caliop_lat))

    # TODO check if caliop datapoints are enveloped by modis pixels
    distances, indices = modis_kdtree.query(caliop_query_points, k=k_neighbors)

    return distances, indices, modis_long[indices], modis_lat[indices]

def plot_collocation(CALIOP_filename, collocation_df, found_MODIS_files):
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
                   label=found_MODIS_files[modis_filename], transform=ccrs.PlateCarree(), c=color)
    
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
                                         delay_minutes=2,
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
    
    stripped_dates, found_MODIS_files = find_MODIS_files_in_interval(MODIS_path, caliop_df.time.iloc[0], caliop_df.time.iloc[-1],
                          delay_minutes,
                          tolerance_minutes)
    number_of_found_MODIS_files = len(found_MODIS_files)

    if number_of_found_MODIS_files == 0:
        return "no MODIS files found in time interval"

    if number_of_found_MODIS_files > 2:
        return "too many MODIS files found"

    MODIS_reader = SD(os.path.join(MODIS_path, found_MODIS_files[0]))
    modis_long = MODIS_reader.select("Longitude").get()
    modis_lat = MODIS_reader.select("Latitude").get()
    number_of_pixels_in_first_file = np.size(modis_long)

    if number_of_found_MODIS_files == 2:
        MODIS_reader = SD(os.path.join(MODIS_path, found_MODIS_files[1]))
        modis_long = np.concatenate([modis_long,\
                                    MODIS_reader.select("Longitude").get()], axis=0)
        modis_lat = np.concatenate([modis_lat,\
                                    MODIS_reader.select("Latitude").get()], axis=0)

    # TODO deal with the case of missing MODIS files by calculating the average discrepancy between modis and caliop coords
    distances, caliop_df["modis_idx"], caliop_df["modis_long"], caliop_df["modis_lat"] =\
        collocate_pixels(caliop_df.long, caliop_df.lat, modis_long, modis_lat)
    
    valid_collocations = distances < 1.1 * 5 * 1.4142 / 6371

    caliop_df["MODIS_file"] = np.where(caliop_df.modis_idx < number_of_pixels_in_first_file, 0, 1)
    caliop_df.modis_idx = np.where(caliop_df.modis_idx < number_of_pixels_in_first_file,\
                                   caliop_df.modis_idx, caliop_df.modis_idx - number_of_pixels_in_first_file)
    
    collocation_path = os.path.join("./collocation_database",
                        start_datetime.strftime("%Y"), start_datetime.strftime("%m"))
    
    os.makedirs(collocation_path, exist_ok=True)
    csv_path = os.path.join(collocation_path, csv_name)

    with open(csv_path, 'w') as file:
        file.write(caliop_df.time.iloc[0].strftime("%H:%M:%S") + " " + \
                   caliop_df.time.iloc[-1].strftime("%H:%M:%S") + "\n")
        file.writelines([modis_file + " " for modis_file in stripped_dates])
        file.write("\n\n")

    caliop_df[["modis_idx", "MODIS_file"]].to_csv(csv_path, mode='a')

    if save_img:
        plot_collocation(csv_name[0:-4], caliop_df, stripped_dates)

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