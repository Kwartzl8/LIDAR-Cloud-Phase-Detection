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
import subprocess
import tempfile as tmp
import warnings

from merge_collocations import merge_collocation_data

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
        
        # get the year and month of the modis file to add to the search strings
        year_month = str(modis_start_time.year) + '/' + str(modis_start_time.month).zfill(2)
        found_files = glob.glob(f"{year_month}/**/*" + search_strings[i] + "*.hdf", root_dir=folder_to_search, recursive=True)

        if len(found_files) > 1:
            # something went terribly wrong
            # TODO have the case where no files are found or when multiples files are found
            print(f"Found {len(found_files)} files for search string {search_strings[i]}")
            continue

        if len(found_files) == 0:
            continue
        
        filename_list[i] = found_files[0]
    
    filename_list = [filename for filename in filename_list if filename != '']

    return search_strings, filename_list

def collocate_pixels(caliop_long, caliop_lat, modis_long, modis_lat, k_neighbors=1):
    modis_long = modis_long.flatten()
    modis_lat = modis_lat.flatten()

    modis_kdtree = KDTree(list(zip(modis_long * np.cos(np.pi * modis_lat / 180), modis_lat)))
    caliop_query_points = list(zip(caliop_long * np.cos(np.pi * caliop_lat / 180), caliop_lat))

    # TODO check if caliop datapoints are enveloped by modis pixels
    distances, indices = modis_kdtree.query(caliop_query_points, k=k_neighbors)

    # convert distances from degrees to radians before returning
    return np.pi * distances / 180, indices, modis_long[indices], modis_lat[indices]

def plot_collocation(CALIOP_filename, collocation_df, found_MODIS_files):
    MODIS_filenames = collocation_df.MODIS_file.unique()
    number_of_found_MODIS_files = len(MODIS_filenames)

    caliop_central_long = (collocation_df.long.max() + collocation_df.long.min())/2
    caliop_central_lat = (collocation_df.lat.max() + collocation_df.lat.min())/2
    ccrs_projection = ccrs.Orthographic(central_longitude=caliop_central_long, central_latitude=caliop_central_lat)
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': ccrs_projection})
    ax.set_extent([collocation_df.long.min(), collocation_df.long.max(), collocation_df.lat.min(), collocation_df.lat.max()], ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=1)

    colors = ["darkblue", "orange", "green"]

    for index, modis_filename in enumerate(MODIS_filenames):
        color = colors[index]
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

def read_JASMIN_CALIOP_file(CALIOP_file):
# I will attempt to copy the file first because the connection is wonky. If it times out, I move on.
    # def copy_file_to_current_directory(source_path):
    #     filename = os.path.basename(source_path)
    #     temp_path = os.path.join(tmp.gettempdir(), filename)
    #     command = f"cp {os.path.abspath(source_path)} {temp_path}"
    #     try:
    #         subprocess.run(command.split(' '), timeout=10)
    #         # print(f"File '{filename}' copied to a temporary directory.")
    #         return temp_path
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         return -1
    
    # temp_path = copy_file_to_current_directory(CALIOP_file)

    # if temp_path == -1:
    #     return "corrupted"
    
    # CALIOP_file = temp_path

    reader = Caliop_hdf_reader()
    caliop_df = pd.DataFrame(columns=["long", "lat", "time", "profile_id"])

    try:
        caliop_df.long = reader._get_longitude(CALIOP_file)
        caliop_df.lat = reader._get_latitude(CALIOP_file)
        caliop_df.time = reader._get_profile_UTC(CALIOP_file)
        caliop_df.profile_id = reader._get_profile_id(CALIOP_file)
        caliop_df = caliop_df.set_index('profile_id')
    except HDF4Error:
        print("corrupted file, skipping")
        return "corrupted"

    # os.remove(CALIOP_file)

    return caliop_df

def collocate_CALIOP_with_MODIS_in_shape(CALIPSO_file, shape_polygon, csv_name,
                                         MODIS_pre_path="/neodc/modis/data/MYD03/collection61",
                                         delay_minutes=2,
                                         tolerance_minutes=0.5,
                                         save_img=False):
    
    # use a separate function to read the file, sometimes the connection fails
    caliop_df = read_JASMIN_CALIOP_file(CALIPSO_file)

    if type(caliop_df) == str:
        return "corrupted"

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

    MODIS_path = MODIS_pre_path

    # this is specific to the CEDA archive on JASMIN, move to a function? I solved this by using glob on the folder, but it's slow
    # MODIS_path = os.path.join(MODIS_pre_path, start_datetime.strftime("%Y"),\
    #                           start_datetime.strftime("%m"), start_datetime.strftime("%d")
    #                           )
    
    stripped_dates, found_MODIS_files = find_MODIS_files_in_interval(MODIS_path, caliop_df.time.iloc[0], caliop_df.time.iloc[-1],
                          delay_minutes,
                          tolerance_minutes)
    number_of_found_MODIS_files = len(found_MODIS_files)

    if number_of_found_MODIS_files == 0:
        return "no MODIS files found in time interval"

    # given that I checked there is at least one file, read it
    MODIS_reader = SD(os.path.join(MODIS_path, found_MODIS_files[0]))
    modis_long = MODIS_reader.select("Longitude").get()
    modis_lat = MODIS_reader.select("Latitude").get()
    end_pixel_id_in_file = [0]
    end_pixel_id_in_file.append(np.size(modis_long) + end_pixel_id_in_file[-1])

    # now concatenate the rest of the files
    for i in range(1, number_of_found_MODIS_files):
        MODIS_reader = SD(os.path.join(MODIS_path, found_MODIS_files[i]))
        modis_long = np.concatenate([modis_long, MODIS_reader.select("Longitude").get()], axis=0)
        modis_lat = np.concatenate([modis_lat, MODIS_reader.select("Latitude").get()], axis=0)
        end_pixel_id_in_file.append(np.size(modis_long))

    distances, caliop_df["modis_idx"], caliop_df["modis_long"], caliop_df["modis_lat"] =\
        collocate_pixels(caliop_df.long, caliop_df.lat, modis_long, modis_lat)

    # save the index of the MODIS file for each profile based on the pixel index in the end_pixel_id_in_file array
    for i in range(number_of_found_MODIS_files):
        mask = (caliop_df.modis_idx >= end_pixel_id_in_file[i]) & (caliop_df.modis_idx < end_pixel_id_in_file[i+1])
        caliop_df.loc[mask, "MODIS_file"] = i

    caliop_df.MODIS_file = caliop_df.MODIS_file.astype(int)
    end_pixel_id_in_file = np.array(end_pixel_id_in_file)

    # reset the modis_idx to be the index in each separate MODIS file
    caliop_df.modis_idx = caliop_df.modis_idx - end_pixel_id_in_file[caliop_df.MODIS_file.values]

    # deal with the case of missing MODIS files by calculating the average discrepancy between modis and caliop coords
    # keep only valid profiles
    caliop_df = caliop_df[distances < 1.1 * 5 * 1.4142 / 6371]

    # return Nothing found over Greenland if there are less than 10 valid profiles remaining
    if len(caliop_df) < 10:
        # print(f"Less than 10 valid profiles found over Greenland in {CALIPSO_file}.")
        return "nothing found over Greenland"

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

    # JASMIN folder
    # CALIOP_folder = "/gws/nopw/j04/gbov/data/asdc.larc.nasa.gov/data/CALIPSO/LID_L2_05kmMLay-Standard-V4-51/"

    os.makedirs("./collocation_logs", exist_ok=True)

    save_logs_each_iteration = False

    logs_save_time = datetime.datetime.now().strftime("%d_%m_%YT%H_%M_%S")
    collocation_logs_path = "./collocation_logs/collocation_log_" + logs_save_time + ".csv"

    if save_logs_each_iteration:
        with open(collocation_logs_path, mode="w") as f:
            f.write(",output\n")

    greenland_geojson = gpd.read_file("Greenland_ALL.geojson")
    greenland_polygon = greenland_geojson.geometry[1]

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

            for caliop_file_count, CALIOP_file_path in tqdm.tqdm(enumerate(CALIOP_file_list)):
                collocation_outputs[caliop_file_count] = collocate_CALIOP_with_MODIS_in_shape(os.path.join(CALIOP_folder, CALIOP_file_path),
                                                            greenland_polygon,
                                                            CALIOP_file_path[0:-4] + ".csv",
                                                            MODIS_pre_path=MODIS_folder, save_img=False)

                if save_logs_each_iteration:
                    with open(collocation_logs_path, mode="a") as f:
                        f.write(CALIOP_file_path + "," + collocation_outputs[caliop_file_count] + "\n")

            # merge the collocation files of this month
            df = merge_collocation_data(f"./collocation_database/{year}/{month:02d}")
            df.to_csv(f"./collocation_database/{year}/{month:02d}/merged_collocations{year}_{month:02d}.csv", mode='w', index=False)

    # save collocation logs if they have not been saved every iteration
    if not save_logs_each_iteration:
        outputs_df = pd.DataFrame(columns=["caliop_file", "output"])
        outputs_df.caliop_file = CALIOP_file_list
        outputs_df.output = collocation_outputs
        outputs_df.set_index("caliop_file").to_csv(collocation_logs_path)
    
    # save list of corrupted files separately
    corrupted_file_list = CALIOP_file_list[collocation_outputs == "corrupted"]

    if len(corrupted_file_list) == 0:
        return

    if isinstance(corrupted_file_list, str):
        corrupted_file_list = [corrupted_file_list]

    with open("./collocation_logs/corrupted_file_list.csv", 'w') as file:
        file.writelines([corrupted_file + "\n" for corrupted_file in corrupted_file_list])

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
    # default paths in this repo:
    # "./test_data/CALIOP/"
    # "./test_data/MODIS/"
    parser = argparse.ArgumentParser(description="Script for collocating CALIOP with MODIS.")
    parser.add_argument("-year", type=validate_year, nargs="+", help="Years", default=[i for i in range(2006, 2024)])
    parser.add_argument("-month", type=validate_month, nargs="+", help="Months", default=[i for i in range(1, 13)])
    parser.add_argument("-caliopfolder", type=str, help="CALIOP folder", default="/gws/nopw/j04/gbov/data/asdc.larc.nasa.gov/data/CALIPSO/LID_L2_05kmMLay-Standard-V4-51/")
    parser.add_argument("-modisfolder", type=str, help="MODIS folder", default="/neodc/modis/data/MYD35_L2/collection61")

    args = parser.parse_args()
    main(args)