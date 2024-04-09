from pyhdf.SD import *
import pandas as pd
import glob
import os
import numpy as np
import argparse
import tqdm

def get_caliop_id_from_filename(filename):
    return filename[-21:-2]

def get_caliop_cloud_phase(caliop_filepath, **kwargs):
    from caliop import Caliop_hdf_reader
    reader_caliop = Caliop_hdf_reader()
    caliop_df = pd.DataFrame(columns=["profile_id"])
    caliop_df.profile_id = reader_caliop._get_profile_id(caliop_filepath)
    caliop_df = caliop_df.set_index('profile_id')
    tropospheric_aerosol_column_AOD = reader_caliop._get_calipso_data(caliop_filepath, "Column_Optical_Depth_Tropospheric_Aerosols_532")
    stratospheric_aerosol_column_AOD = reader_caliop._get_calipso_data(caliop_filepath, "Column_Optical_Depth_Stratospheric_Aerosols_532")
    total_aerosol_column_AOD = tropospheric_aerosol_column_AOD + stratospheric_aerosol_column_AOD

    # get the CAD score (remove mask)
    CAD_score = reader_caliop._get_calipso_data(caliop_filepath, "CAD_Score").data

    _, layer_type = reader_caliop._get_feature_classification(caliop_filepath, "Feature_Classification_Flags")

    integrated_attenuated_backscatter = reader_caliop._get_calipso_data(caliop_filepath, "Integrated_Attenuated_Backscatter_532").data
    column_cloud_optical_depth = reader_caliop._get_calipso_data(caliop_filepath, "Column_Optical_Depth_Cloud_532").data[0]
    caliop_df['cColumnOpticalDepth'] = column_cloud_optical_depth

    caliop_df["cClear"] = np.where(np.all(layer_type == 1, axis=0), True, False)
    caliop_df["cInvalid"] = np.where(np.any(layer_type == 0, axis=0), True, False)
    caliop_df["cAerosolFree"] = (total_aerosol_column_AOD < 0.05)[0]
    caliop_df.cAerosolFree = caliop_df.cAerosolFree.fillna(True)

    cloud_phase_layer, cloud_phase_layer_qa = reader_caliop._get_cloud_phase(caliop_filepath, "Feature_Classification_Flags")

    # fill out high quality features
    high_quality_cloud_layer_mask = (CAD_score >= 70) & (CAD_score <= 100)
    caliop_df["cCloudyHighQA"] = np.where(np.any(high_quality_cloud_layer_mask, axis=0), True, False)
    high_quality_water_cloud_mask = (cloud_phase_layer == 2) & high_quality_cloud_layer_mask & (cloud_phase_layer_qa == 3)
    high_quality_ice_cloud_mask = (cloud_phase_layer == 1) & high_quality_cloud_layer_mask & (cloud_phase_layer_qa == 3)
    high_quality_oriented_ice_cloud_mask = (cloud_phase_layer == 3) & high_quality_cloud_layer_mask & (cloud_phase_layer_qa == 3)

    caliop_df["cWaterHighQA"] = np.where(np.any(high_quality_water_cloud_mask, axis=0) &\
                                    ~np.any(high_quality_ice_cloud_mask, axis=0) &\
                                    ~np.any(high_quality_oriented_ice_cloud_mask, axis=0), True, False)
    caliop_df["cIceHighQA"] = np.where((np.any(high_quality_ice_cloud_mask, axis=0) |\
                                    np.any(high_quality_oriented_ice_cloud_mask, axis=0)) &\
                                    ~np.any(high_quality_water_cloud_mask, axis=0), True, False)
    caliop_df["cMixedMultilayerHighQA"] = np.where((np.any(high_quality_ice_cloud_mask, axis=0) |\
                                    np.any(high_quality_oriented_ice_cloud_mask, axis=0)) &\
                                    np.any(high_quality_water_cloud_mask, axis=0), True, False)
    caliop_df["cUnknownHighQA"] = np.where(caliop_df["cCloudyHighQA"] &\
                                    ~np.any(high_quality_ice_cloud_mask, axis=0) &\
                                    ~np.any(high_quality_oriented_ice_cloud_mask, axis=0) &\
                                    ~np.any(high_quality_water_cloud_mask, axis=0), True, False)

    # extract some useful classifications from the MixedMultilayer category, now with a more stringent condition
    attenuation_ratio = 5

    IAB_ice = integrated_attenuated_backscatter * (high_quality_ice_cloud_mask | high_quality_oriented_ice_cloud_mask)
    IAB_water = integrated_attenuated_backscatter * high_quality_water_cloud_mask

    caliop_df["cIAB_ice"] = np.sum(IAB_ice, axis=0)
    caliop_df["cIAB_water"] = np.sum(IAB_water, axis=0)

    # add the mixed ice dominant profiles to the ice category
    caliop_df["cIceDominant"] = caliop_df["cIceHighQA"] | np.where(
        caliop_df["cMixedMultilayerHighQA"] &\
        (caliop_df["cIAB_ice"] >\
        caliop_df["cIAB_water"] * attenuation_ratio), True, False
    )

    # add the mixed water dominant profiles to the ice category
    caliop_df["cWaterDominant"] = caliop_df["cWaterHighQA"] | np.where(
        caliop_df["cMixedMultilayerHighQA"] &\
        (caliop_df["cIAB_ice"] <\
        caliop_df["cIAB_water"] / attenuation_ratio), True, False
        )
    
    # # now these are the truly mixed profiles
    # caliop_df["cAmbiguous"] = np.where(
    #     caliop_df["cMixedMultilayerHighQA"] &\
    #     ((caliop_df["cIAB_ice"] >\
    #     caliop_df["cIAB_water"] / attenuation_ratio) |\
    #     (caliop_df["cIAB_ice"] <\
    #     caliop_df["cIAB_water"] * attenuation_ratio)), True, False
    # )

    caliop_df["cAmbiguous"] = np.where(
        caliop_df["cMixedMultilayerHighQA"] &\
        ~caliop_df["cIceDominant"] &\
        ~caliop_df["cWaterDominant"], True, False
    )

    # get the cloud geometrical thickness
    layer_top_altitude = reader_caliop._get_calipso_data(caliop_filepath, "Layer_Top_Altitude").data
    layer_base_altitude = reader_caliop._get_calipso_data(caliop_filepath, "Layer_Base_Altitude").data
    column_ice_cloud_geometrical_thickness = np.sum((layer_top_altitude - layer_base_altitude) *\
                                                (high_quality_ice_cloud_mask | high_quality_oriented_ice_cloud_mask), axis=0)
    column_water_cloud_geometrical_thickness = np.sum((layer_top_altitude - layer_base_altitude) * high_quality_water_cloud_mask, axis=0)

    caliop_df['cCloudGeometricalThickness'] = column_ice_cloud_geometrical_thickness * caliop_df['cIceDominant'].astype(int) +\
          column_water_cloud_geometrical_thickness * caliop_df['cWaterDominant'].astype(int)

    # get cloud top height for each category
    ice_cloud_top_height = np.max(layer_top_altitude * (high_quality_ice_cloud_mask | high_quality_oriented_ice_cloud_mask), axis=0)
    water_cloud_top_height = np.max(layer_top_altitude * high_quality_water_cloud_mask, axis=0)

    # combine them
    caliop_df['cCloudTopHeight'] = ice_cloud_top_height * caliop_df['cIceDominant'].astype(int) +\
          water_cloud_top_height * caliop_df['cWaterDominant'].astype(int)



    return caliop_df

def downsample_array_by_mode(input_array, pixelXsize=5, pixelYsize=5):
    input_array = np.array(input_array)
    rows, cols = input_array.shape
    
    # reduce array to reshapeable size
    rows, cols = rows//pixelXsize * pixelXsize, cols//pixelYsize * pixelYsize
    input_array = input_array[:rows, :cols]

    # the dimensions of the output array
    rows, cols = rows//pixelXsize, cols//pixelYsize
    
    # reshape to 3d array where each axb pixel contains an array of a*b subpixels
    reshaped_array = input_array.reshape(rows, pixelXsize, cols, pixelYsize).transpose((0, 2, 1, 3)).reshape(rows, cols, pixelXsize*pixelYsize)

    # output_array = scipy.stats.mode(reshaped_array, axis=2, keepdims=True)[0]
    output_array = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=2, arr=reshaped_array)

    return output_array

def get_modis_5km_cloud_phase(modis_filepath, daytime_only=False, **kwargs):
    if not os.path.exists(modis_filepath):
        print("The given MODIS file does not exist.")
        return

    df = pd.DataFrame(columns=["pixel_id", "mCloudy", "mClear", "mWater", "mIce", "mUndetermined"])

    reader = SD(modis_filepath)
    if not daytime_only:
        phase_1km  = reader.select("Cloud_Phase_Infrared_1km").get()
        fill_value = reader.select("Cloud_Phase_Infrared_1km").attributes().get("_FillValue")
        phase_1km = np.where((phase_1km == fill_value) | (phase_1km == 6), 3, phase_1km)
    else:
        phase_1km  = reader.select("Cloud_Phase_Optical_Properties").get()
        fill_value = reader.select("Cloud_Phase_Optical_Properties").attributes().get("_FillValue")
        phase_1km = np.where(phase_1km == fill_value, 5, phase_1km)
        phase_1km -= 1

    phase_downsampled = phase_1km[::5, ::5][:, :-1].flatten()

    df["pixel_id"] = np.arange(len(phase_downsampled))
    df.set_index("pixel_id", inplace=True)
    df["mClear"] = phase_downsampled == 0
    df["mWater"] = phase_downsampled == 1
    df["mIce"] = phase_downsampled == 2
    df["mUndetermined"] = phase_downsampled == 3
    df["mDayPixel"] = reader.select("Solar_Zenith").get().flatten() < 8136

    if daytime_only:
        df["mOptical"] = phase_downsampled != 4

    # should I set the cloudy flags according to the mask or according to the phase product?
    # cloud_mask_quality_flag = (MODIS_reader.select("Cloud_Mask_5km").get()[:,:,0] >> 1) & 0b11
    # df["mCloudy"] = (np.array(cloud_mask_quality_flag).flatten() == 0) | (np.array(cloud_mask_quality_flag).flatten() == 1)

    df["mCloudy"] = df["mWater"] | df["mIce"] | df["mUndetermined"]

    return df

def get_modis_5km_optical_cloud_phase(modis_filepath, **kwargs):
    return get_modis_5km_cloud_phase(modis_filepath, daytime_only=True, **kwargs)

def get_modis_l1_data(filepath, **kwargs):
    # parameter validation
    if not os.path.exists(filepath):
        print("The given MODIS file does not exist.")
        return pd.DataFrame()
    
    # try to open the file
    try:
        reader = SD(filepath)
    except HDF4Error:
        print("The given MODIS file could not be opened.")
        return

    # # check if **kwargs contains the variables_to_extract key. If it does, save the value in variables_to_extract, otherwise, use the default values
    # if "variables_to_extract" in kwargs:
    #     variables_to_extract = kwargs["variables_to_extract"]
    # else:
    variables_to_extract = ["EV_250_Aggr1km_RefSB", "EV_500_Aggr1km_RefSB", "EV_1KM_RefSB", "EV_1KM_Emissive"]

    data = [0] * len(variables_to_extract)
    data_masks = [0] * len(variables_to_extract)
    band_names = []
    for index, variable in enumerate(variables_to_extract):
        try:
            extracted_data = reader.select(variable).get()
        except HDF4Error:
            print(f"The variable {variable} could not be found in the {os.path.basename(filepath)} file.")
            return pd.DataFrame()

        # subsample every 5th pixel in the x and y direction and get rid of the last column
        extracted_data = extracted_data[:, 2::5, 2::5][:, :, :-1]

        # get valid range for reflectance from hdf file
        valid_range = reader.select(variable).attributes().get("valid_range")

        # replace all values of 65533 (saturated detector) with the maximum value allowed in valid_range
        extracted_data = np.where(extracted_data == 65533, valid_range[1], extracted_data)

        # replace reflectances with a masked array where values are not in range
        extracted_data_mask = (extracted_data < valid_range[0]) | (extracted_data > valid_range[1])

        # get radiance scales and offsets for both datasets from hdf file
        radiance_scales = reader.select(variable).attributes().get("radiance_scales")
        radiance_offsets = reader.select(variable).attributes().get("radiance_offsets")

        # broadcast the radiance scales and offsets to the shape of the reflectances array
        radiance_scales = np.tile(radiance_scales, (extracted_data.shape[1], extracted_data.shape[2], 1)).transpose((2, 0, 1))
        radiance_offsets = np.tile(radiance_offsets, (extracted_data.shape[1], extracted_data.shape[2], 1)).transpose((2, 0, 1))

        # apply radiance scales and offsets to reflectances
        extracted_data = radiance_scales * (extracted_data - radiance_offsets)

        data[index] = extracted_data
        data_masks[index] = extracted_data_mask
        band_names.append(reader.select(variable).attributes().get("band_names").split(","))

    data = np.concatenate(data, axis=0)
    data_masks = np.concatenate(data_masks, axis=0)

    data = np.ma.masked_array(data, data_masks)
    # flatten the band names list
    band_names = ["band_" + item for sublist in band_names for item in sublist]

    # get pixel id
    pixel_id = pd.Series(np.arange(data.shape[1] * data.shape[2]), name="pixel_id")

    # save the data into a dataframe, with the band names as column names, and the rows as the masked values in data
    df = pd.DataFrame(data.reshape((data.shape[0], data.shape[1] * data.shape[2])).transpose(), columns=band_names, index=pixel_id)

    # remove empty columns
    df = df.dropna(axis=1, how='all')

    return df

def get_snow_ice_cover_classification(L3_land_cover_folder, longitudes, latitudes, year_list):
    # check whether L3_land_cover_folder exists. If not throw an error
    if not os.path.exists(L3_land_cover_folder):
        raise FileNotFoundError("The given L3 land cover folder does not exist.")


    # check whether year_list is a scalar and if it is, put it into a list the size of lats and longs
    if np.isscalar(year_list):
        year_list = np.full(np.shape(longitudes), year_list)

    snow_ice_cover = np.full(np.shape(longitudes), -1)
    unique_year_list = np.unique(year_list)

    for year in unique_year_list:
        # get the file path
        search_string = f"**/MCD12C1.A{year}*"
        file_path = glob.glob(search_string, root_dir=L3_land_cover_folder, recursive=True)

        # check if the file exists
        if len(file_path) == 0:
            print(f"File for year {year} not found.")
            continue
        
        # check if there are multiple files for the same year
        if len(file_path) > 1:
            print(f"Multiple files for year {year} found.")
            continue
        

        # complete filepath
        file_path = os.path.join(L3_land_cover_folder, file_path[0])
        # print(file_path)

        # open the file
        reader = SD(file_path)

        # get the IGBP majority land cover type
        majority_land_cover_type = reader.select("Majority_Land_Cover_Type_1").get()

        # round down the longitudes and latitudes in the current year to the nearest 0.05 and convert to integers by multiplying by 20
        longitudes_rounded = np.floor(longitudes[(year_list == year)] / 0.05) + 3600
        latitudes_rounded = - np.floor(latitudes[(year_list == year)] / 0.05) + 1800 - 1

        # if any values are -1, replace them with 0
        longitudes_rounded[longitudes_rounded == 7200] = 7199
        latitudes_rounded[latitudes_rounded == -1] = 0

        # get a true or false value for snow/ice cover at the rounded longitudes and latitudes
        yearly_snow_ice_cover = majority_land_cover_type[latitudes_rounded.astype(int), longitudes_rounded.astype(int)] == 15

        # add the yearly snow/ice cover to the total snow/ice cover
        snow_ice_cover[year_list == year] = yearly_snow_ice_cover

    # if there are no -1 values left in the snow_ice_cover array, convert it to a boolean array. If there are, print a warning message
    if -1 not in snow_ice_cover:
        snow_ice_cover = snow_ice_cover.astype(bool)
    else:
        print("There are still -1 values in the snow_ice_cover array. The snow/ice cover classification may be incomplete.")

    return snow_ice_cover

def modis_fileID_to_datetime(modis_fileID):
    # mFileID example: "A2017034.1255", where A is the satellite identifier, 2017 is the year, 034 is the day of the year, and 1255 is the time of the day in minutes
    year = modis_fileID[1:5]
    day_of_year = modis_fileID[5:8]
    time = modis_fileID[9:13]
    return pd.to_datetime(year + day_of_year + time, format="%Y%j%H%M")

def datetime_to_modis_fileID(datetime):
    return "A" + datetime.strftime("%Y%j.%H%M")

def forward_mapping(long, lat):
    # constants
    R = 6371007.181     # radius of the Earth in meters
    T = 1111950         # the height and width of each MODIS tile in the projection plane
    xmin = -20015109    # the western limit of the projection plane
    ymax = 10007555     # the northern limit of the projection plane
    w = T / 1200        # the actual size of a 1km MODIS sinusoidal grid cell

    # calculate the x and y coordinates
    x = R * np.radians(long) * np.cos(np.radians(lat))
    y = R * np.radians(lat)

    # compute the horizontal (H) and vertical (V) tile coordinates
    H = np.floor((x - xmin) / T).astype(int)
    V = np.floor((ymax - y) / T).astype(int)

    # calculate the row and column indices
    i = np.floor(((ymax - y) % T)/ w).astype(int)
    j = np.floor(((x - xmin) % T)/ w).astype(int)

    return pd.DataFrame({"H": H, "V": V, "i": i, "j": j})

def get_myd11a1_surface_temp(MYD11A1_folder, long, lat, L2_fileID):
    # throw out time information, I only need the day of the year
    MYD11A1_fileID = L2_fileID[:-5]

    datetime = modis_fileID_to_datetime(L2_fileID)

    # add year, month to the folder path for faster search
    year_month = datetime.strftime("%Y/%m")
    MYD11A1_folder = os.path.join(MYD11A1_folder, year_month)
    # check whether MYD11A1_folder exists. If not throw an error
    if not os.path.exists(MYD11A1_folder):
        raise FileNotFoundError("The given MYD11A1 folder does not exist.")

    tiles_and_indices = forward_mapping(long, lat)
    tiles_to_pixel_id_dict = tiles_and_indices.groupby(["H", "V"]).groups

    output_df = pd.DataFrame(columns=["mLST_day", "mLST_night", "pixel_id"])
    for HV_tile, pixel_ids_in_tile in tiles_to_pixel_id_dict.items():
        # read the MYD11A1 file
        search_string = f"**/MYD11A1.{MYD11A1_fileID}.h{HV_tile[0]:02}v{HV_tile[1]:02}*.hdf"
        file_path = glob.glob(search_string, root_dir=MYD11A1_folder, recursive=True)

        # check if the file exists
        if len(file_path) == 0:
            print(f"MYD11A1 file with ID {MYD11A1_fileID}.h{HV_tile[0]:02}v{HV_tile[1]:02} not found.")
            output_df = pd.concat([output_df, pd.DataFrame({"mLST_day": np.full(len(pixel_ids_in_tile), np.nan), "mLST_night": np.full(len(pixel_ids_in_tile), np.nan), "pixel_id": pixel_ids_in_tile})])
            continue

        # check if there are multiple files for the same day
        if len(file_path) > 1:
            print(f"Multiple MYD11A1 files with ID {MYD11A1_fileID} found.")
            continue

        # complete filepath
        file_path = os.path.join(MYD11A1_folder, file_path[0])

        # open the file
        reader = SD(file_path)
        LST_day = reader.select("LST_Day_1km").get()
        LST_night = reader.select("LST_Night_1km").get()

        # get valid range for LST from hdf file
        valid_range = reader.select("LST_Day_1km").attributes().get("valid_range")

        # replace LST with a masked array where values are not in range
        LST_day = np.ma.masked_where((LST_day < valid_range[0]) | (LST_day > valid_range[1]), LST_day)
        LST_night = np.ma.masked_where((LST_night < valid_range[0]) | (LST_night > valid_range[1]), LST_night)

        # get scale factor
        scale_factor_day = reader.select("LST_Day_1km").attributes().get("scale_factor")
        scale_factor_night = reader.select("LST_Night_1km").attributes().get("scale_factor")

        # apply scale factor
        LST_day = scale_factor_day * LST_day
        LST_night = scale_factor_night * LST_night

        # # plot the LST
        # import matplotlib.pyplot as plt
        # plt.matshow(LST_day)
        # plt.gca().set_title(f"LST Day, {datetime.strftime('%Y-%m-%d'), {HV_tile}}")
        # plt.colorbar()
        # plt.show()

        # plt.matshow(LST_night)
        # plt.gca().set_title(f"LST Night, {datetime.strftime('%Y-%m-%d'), {HV_tile}}")
        # plt.colorbar()
        # plt.show()

        # only keep data at the pixel_ids_in_tile
        LST_day = LST_day[tiles_and_indices.loc[pixel_ids_in_tile, "i"], tiles_and_indices.loc[pixel_ids_in_tile, "j"]].flatten()
        LST_night = LST_night[tiles_and_indices.loc[pixel_ids_in_tile, "i"], tiles_and_indices.loc[pixel_ids_in_tile, "j"]].flatten()

        # add the data to the output dataframe
        output_df = pd.concat([output_df, pd.DataFrame({"mLST_day": LST_day, "mLST_night": LST_night, "pixel_id": pixel_ids_in_tile})])
    
    return output_df.sort_values(by="pixel_id").drop(columns=["pixel_id"])

def get_myd11a2_surface_temp(MYD11A2_folder, long, lat, L2_fileID):
    year = L2_fileID[1:5]
    day_of_year = int(L2_fileID[5:8])
    time = L2_fileID[9:13]

    # floor the day of the year to a multiple of 8 + 1
    multiple_of_8 = ((day_of_year - 1) // 8) * 8 + 1
    datetime = pd.to_datetime(year + str(multiple_of_8).zfill(3) + time, format="%Y%j%H%M")
    MYD11A2_fileID = datetime_to_modis_fileID(datetime)
    # throw out time information, I only need the day of the year
    MYD11A2_fileID = MYD11A2_fileID[:-5]

    # add year, month to the folder path for faster search
    year_month = datetime.strftime("%Y/%m")
    MYD11A2_folder = os.path.join(MYD11A2_folder, year_month)
    # check whether MYD11A2_folder exists. If not throw an error
    if not os.path.exists(MYD11A2_folder):
        raise FileNotFoundError("The given MYD11A1 folder does not exist.")

    tiles_and_indices = forward_mapping(long, lat)
    tiles_to_pixel_id_dict = tiles_and_indices.groupby(["H", "V"]).groups

    output_df = pd.DataFrame(columns=["mLST_day", "mLST_night", "pixel_id"])
    for HV_tile, pixel_ids_in_tile in tiles_to_pixel_id_dict.items():
        # read the MYD11A1 file
        search_string = f"**/MYD11A2.{MYD11A2_fileID}.h{HV_tile[0]:02}v{HV_tile[1]:02}*.hdf"
        file_path = glob.glob(search_string, root_dir=MYD11A2_folder, recursive=True)

        # check if the file exists
        if len(file_path) == 0:
            print(f"MYD11A1 file with ID {MYD11A2_fileID}.h{HV_tile[0]:02}v{HV_tile[1]:02} not found.")
            output_df = pd.concat([output_df, pd.DataFrame({"mLST_day": np.full(len(pixel_ids_in_tile), np.nan), "mLST_night": np.full(len(pixel_ids_in_tile), np.nan), "pixel_id": pixel_ids_in_tile})])
            continue

        # check if there are multiple files for the same day
        if len(file_path) > 1:
            print(f"Multiple MYD11A2 files with ID {MYD11A2_fileID} found.")
            continue

        # complete filepath
        file_path = os.path.join(MYD11A2_folder, file_path[0])

        # open the file
        reader = SD(file_path)
        LST_day = reader.select("LST_Day_1km").get()
        LST_night = reader.select("LST_Night_1km").get()

        # get valid range for LST from hdf file
        valid_range = reader.select("LST_Day_1km").attributes().get("valid_range")

        # replace LST with a masked array where values are not in range
        LST_day = np.ma.masked_where((LST_day < valid_range[0]) | (LST_day > valid_range[1]), LST_day)
        LST_night = np.ma.masked_where((LST_night < valid_range[0]) | (LST_night > valid_range[1]), LST_night)

        # get scale factor
        scale_factor_day = reader.select("LST_Day_1km").attributes().get("scale_factor")
        scale_factor_night = reader.select("LST_Night_1km").attributes().get("scale_factor")

        # apply scale factor
        LST_day = scale_factor_day * LST_day
        LST_night = scale_factor_night * LST_night

        # # plot the LST
        # import matplotlib.pyplot as plt
        # plt.matshow(LST_day)
        # plt.gca().set_title(f"LST Day, {datetime.strftime('%Y-%m-%d'), {HV_tile}}")
        # plt.colorbar()
        # plt.show()

        # plt.matshow(LST_night)
        # plt.gca().set_title(f"LST Night, {datetime.strftime('%Y-%m-%d'), {HV_tile}}")
        # plt.colorbar()
        # plt.show()

        # only keep data at the pixel_ids_in_tile
        LST_day = LST_day[tiles_and_indices.loc[pixel_ids_in_tile, "i"], tiles_and_indices.loc[pixel_ids_in_tile, "j"]].flatten()
        LST_night = LST_night[tiles_and_indices.loc[pixel_ids_in_tile, "i"], tiles_and_indices.loc[pixel_ids_in_tile, "j"]].flatten()

        # add the data to the output dataframe
        output_df = pd.concat([output_df, pd.DataFrame({"mLST_day": LST_day, "mLST_night": LST_night, "pixel_id": pixel_ids_in_tile})])
    
    return output_df.sort_values(by="pixel_id").set_index("pixel_id")

def get_modis_geo_data(filepath, **kwargs):
    # parameter validation
    if not os.path.exists(filepath):
        print("The given MODIS file does not exist.")
        return
    
    # try to open the file
    try:
        reader = SD(filepath)
    except HDF4Error:
        print("The given MODIS file could not be opened.")
        return
    
    # get latitude and longitude
    latitude = reader.select("Latitude").get()
    longitude = reader.select("Longitude").get()

    # replace latitude and longitude with a masked array where values are not in range
    latitude = np.ma.masked_where((latitude < -90) | (latitude > 90), latitude)
    longitude = np.ma.masked_where((longitude < -180) | (longitude > 180), longitude)

    # get the year string from the file name MYD06_L2.AYYYYDDD.HHMM.CCC.YYYYDDDHHMMSS.hdf)
    #                                        0123456789
    # string is from 10 to 13
    year_list = np.full(latitude.shape, int(os.path.basename(filepath)[10:14]))

    # get view zenith angle
    view_zenith_angle = reader.select("Sensor_Zenith").get() / 100

    # get solar zenith angle
    solar_zenith_angle = reader.select("Solar_Zenith").get() / 100

    # get pixel id
    pixel_id = pd.Series(np.arange(np.size(latitude)), name="pixel_id")

    # save the data into a dataframe with the column names as "mLong", "mLat", "mVZA", "mSZA"
    df = pd.DataFrame(np.array([longitude.flatten(), latitude.flatten(), view_zenith_angle.flatten(), solar_zenith_angle.flatten()]).transpose(), columns=["mLong", "mLat", "mVZA", "mSZA"], index=pixel_id)
    
    if kwargs is None:
        kwargs = {"kwargs":{}}
    
    # check if **kwargs contains the surface_datapath key. If it does, save the value in surface_product_path
    if "surface_datapath" in kwargs["kwargs"]:
        surface_product_path = kwargs["kwargs"]["surface_datapath"]
        snow_ice_cover = get_snow_ice_cover_classification(surface_product_path, longitude, latitude, year_list)
        df["mSnowIceCover"] = snow_ice_cover.flatten()

    if "myd11_path" in kwargs["kwargs"]:
        myd11_path = kwargs["kwargs"]["myd11_path"]
        fileID = os.path.basename(filepath)[9:22]
        surface_temp_df = get_myd11a2_surface_temp(myd11_path, longitude.flatten(), latitude.flatten(), fileID)

        df = pd.concat([df, surface_temp_df], axis=1)

    return df

def get_modis_cloud_top_properties(filepath, **kwargs):
    # check whether the file exists
    if not os.path.exists(filepath):
        print("The given MODIS file does not exist.")
        return 
    # try to open the file
    try:
        reader = SD(filepath)
    except HDF4Error:
        print("The given MODIS file could not be opened.")
        return
    
    # get the cloud top temperature
    cloud_top_temp = reader.select("Cloud_Top_Temperature").get()

    # get valid range for cloud top temperature from hdf file
    valid_range = reader.select("Cloud_Top_Temperature").attributes().get("valid_range")

    # replace cloud top temperature with a masked array where values are not in range
    cloud_top_temp = np.ma.masked_where((cloud_top_temp < valid_range[0]) | (cloud_top_temp > valid_range[1]), cloud_top_temp)

    # get scale factor
    scale_factor = reader.select("Cloud_Top_Temperature").attributes().get("scale_factor")

    # get offset
    offset = reader.select("Cloud_Top_Temperature").attributes().get("add_offset")

    # apply scale factor and offset
    cloud_top_temp = scale_factor * (cloud_top_temp - offset)

    # get cloud top pressure
    cloud_top_pressure = reader.select("Cloud_Top_Pressure_Infrared").get()

    # get valid range for cloud top pressure from hdf file
    valid_range = reader.select("Cloud_Top_Pressure_Infrared").attributes().get("valid_range")

    # replace cloud top pressure with a masked array where values are not in range
    cloud_top_pressure = np.ma.masked_where((cloud_top_pressure < valid_range[0]) | (cloud_top_pressure > valid_range[1]), cloud_top_pressure)

    # get scale factor (the offset is 0 for this variable)
    scale_factor = reader.select("Cloud_Top_Pressure_Infrared").attributes().get("scale_factor")

    # apply scale factor
    cloud_top_pressure = cloud_top_pressure * scale_factor

    # now get emissivity data
    emissivity_dataset_names = [
        "cloud_emiss11_1km",
        "cloud_emiss12_1km",
        "cloud_emiss85_1km",
    ]

    emissivity_data = [0] * len(emissivity_dataset_names)

    for index, emissivity_dataset_name in enumerate(emissivity_dataset_names):
        emissivity_data[index] = reader.select(emissivity_dataset_name).get()[2::5, 2::5][:, :-1]

        # get valid range for emissivity from hdf file
        valid_range = reader.select(emissivity_dataset_name).attributes().get("valid_range")

        # replace emissivity with a masked array where values are not in range
        emissivity_data[index] = np.ma.masked_where((emissivity_data[index] < valid_range[0]) | (emissivity_data[index] > valid_range[1]), emissivity_data[index])

        # get scale factor and offset
        scale_factor = reader.select(emissivity_dataset_name).attributes().get("scale_factor")
        offset = reader.select(emissivity_dataset_name).attributes().get("add_offset")

        # apply scale factor and offset
        emissivity_data[index] = scale_factor * (emissivity_data[index] - offset)

    emissivities = dict(zip(emissivity_dataset_names, emissivity_data))

    # calculate beta parameters
    beta_11_12 = np.log(1 - emissivities["cloud_emiss11_1km"]) / np.log(1 - emissivities["cloud_emiss12_1km"])
    beta_85_11 = np.log(1 - emissivities["cloud_emiss85_1km"]) / np.log(1 - emissivities["cloud_emiss11_1km"])

    # get pixel id
    pixel_id = pd.Series(np.arange(cloud_top_temp.shape[0] * cloud_top_temp.shape[1]), name="pixel_id")

    # save the data into a dataframe, with column names "mCloudTopTemp", "mCloudTopPressure", making sure only the masked data is kept
    df = pd.DataFrame(np.ma.array([cloud_top_temp.flatten(), cloud_top_pressure.flatten(), beta_11_12.flatten(), beta_85_11.flatten()]).transpose(), columns=["mCloudTopTemp", "mCloudTopPressure", "mBeta11_12", "mBeta85_11"], index=pixel_id)

    return df

# downsampling function that takes an array and downsamples it to a given pixel size by taking the average of the subpixels, while ignoring the masked values
def downsample_masked_array_by_mean(input_masked_array, pixelXsize=5, pixelYsize=5):
    rows, cols = input_masked_array.shape

    # reduce array to reshapeable size
    rows, cols = rows//pixelXsize * pixelXsize, cols//pixelYsize * pixelYsize
    input_masked_array = input_masked_array[:rows, :cols]

    # the dimensions of the output array
    rows, cols = rows//pixelXsize, cols//pixelYsize
    
    # reshape to 3d array where each axb pixel contains an array of a*b subpixels
    reshaped_array = np.ma.reshape(input_masked_array, (rows, pixelXsize, cols, pixelYsize)).transpose((0, 2, 1, 3)).reshape(rows, cols, pixelXsize*pixelYsize)

    # take mean while ignoring masked values
    output_array = np.ma.mean(reshaped_array, axis=2)
    
    return output_array

def get_modis_surface_temperature(filepath, **kwargs):
    # check whether the file exists
    if not os.path.exists(filepath):
        print("The given MODIS file does not exist.")
        return 
    # try to open the file
    try:
        reader = SD(filepath)
    except HDF4Error:
        print("The given MODIS file could not be opened.")
        return
    
    # get the land surface temperature (LST)
    surface_temp = reader.select("LST").get()

    # get valid range for surface temperature from hdf file
    valid_range = reader.select("LST").attributes().get("valid_range")

    # replace surface temperature with a masked array where values are not in range
    surface_temp = np.ma.masked_where((surface_temp < valid_range[0]) | (surface_temp > valid_range[1]), surface_temp)

    # get scale factor
    scale_factor = reader.select("LST").attributes().get("scale_factor")

    # apply scale factor
    surface_temp = surface_temp * scale_factor
    
    # downsample to 5km resolution
    surface_temp_downsampled = downsample_masked_array_by_mean(surface_temp, 5, 5)

    pixel_id = pd.Series(np.arange(surface_temp_downsampled.shape[0] * surface_temp_downsampled.shape[1]), name="pixel_id")

    df = pd.DataFrame(surface_temp_downsampled.flatten(), columns=["surface_temp"], index=pixel_id)
    return df

def get_feature_sets(feature_set, data_folder, collocation_data_path, **kwargs):
    # define available functions and corresponding variables (TODO store in a text file)
    existing_feature_sets = ["modis_infrared_cloud_phase", "modis_optical_cloud_phase", "modis_radiances", "modis_geo", "modis_cloud_top_properties", "modis_surface_temp", "caliop_cloud_phase"]

    feature_set_extractor_func_dict = dict(zip(existing_feature_sets,\
        [get_modis_5km_cloud_phase, get_modis_5km_optical_cloud_phase, get_modis_l1_data,\
         get_modis_geo_data, get_modis_cloud_top_properties, get_modis_surface_temperature, get_caliop_cloud_phase]))
    product_name_dict = dict(zip(existing_feature_sets, ["MYD06", "MYD06", "MYD021KM", "MYD06", "MYD06", "MYD11", "CAL_LID_L2_05kmMLay-Standard-V4-51"]))
    fileID_keys_dict = dict(zip(existing_feature_sets, ["mFileID", "mFileID", "mFileID", "mFileID", "mFileID", "mFileID", "cFileID"]))
    profileID_keys_dict = dict(zip(existing_feature_sets, ["pixel_id", "pixel_id", "pixel_id", "pixel_id", "pixel_id", "pixel_id", "profile_id"]))

    # parameter validation
    if feature_set not in existing_feature_sets:
        print(f"{feature_set} not in the existing feature set extractor functions. The parameter feature_set should be one of the following:")
        print(", ".join(existing_feature_sets))
        return
    
    # read in the collocation file
    collocation_df = pd.read_csv(collocation_data_path).sort_values(by=["cFileID", "profile_id"])
    collocated_file_ids = collocation_df[fileID_keys_dict[feature_set]]
    collocated_profile_ids = collocation_df[profileID_keys_dict[feature_set]]
    collocated_data_df_list = []

    # loop over unique file ids
    for file_id in tqdm.tqdm(collocated_file_ids.unique()):
        # to speed up the file search, add the year and month to the search string
        year_month = collocation_df[collocation_df[fileID_keys_dict[feature_set]] == file_id].iloc[0]["cFileID"][:7].replace("-", "/")
        file_path = glob.glob(f"{year_month}/**/{product_name_dict[feature_set]}*{file_id}*", root_dir=data_folder, recursive=True)

        if len(file_path) == 0:
            print(file_id, "file not found")
            continue
        
        file_path = os.path.join(data_folder, file_path[0])
        
        # call the desired function
        df = feature_set_extractor_func_dict[feature_set](file_path, **kwargs)

        if df.empty:
            continue

        # reduce to collocated pixels
        collocated_profile_ids_in_file = collocated_profile_ids[collocated_file_ids == file_id]
        df = df.loc[collocated_profile_ids_in_file, :]

        # add the file id to uniquely identify each profile
        df["fileID"] = [file_id] * len(df)
        df.set_index(["fileID"], append=True, inplace=True)

        collocated_data_df_list.append(df)

    return pd.concat(collocated_data_df_list)

def main(args):
    if args.s == "?":
        print("Give to the parameter -s the name of the file the feature set should be saved in.")
        return

    # if there is no attribute kwargs, set it to an empty dictionary
    if "kwargs" not in args:
        args.kwargs = {}

    df = get_feature_sets(args.f, args.datapath, args.colpath, **args.kwargs)

    # convert any boolean values to int, and limit the significant digits of floating point values to 4
    df = df.round(4)
    df = df.replace({True: 1, False: 0})

    # save the dataframe where the collocation data is
    df.to_csv(os.path.join(os.path.dirname(args.colpath), args.s))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for extracting feature sets.")
    parser.add_argument("-f", help="The name of the feature set to be extracted", type=str)
    parser.add_argument("-datapath", help="The path to the folder of the relevant products", type=str)
    parser.add_argument("-colpath", help="The path to the collocation file", type=str)
    parser.add_argument("-s", type=str, help="The name of the file to be saved (will be saved in the same folder as the collocation file)", default="?")
    parser.add_argument("-kwargs", type=dict, help="Optional arguments for the feature set extractor function", default={})

    args = parser.parse_args()
    main(args)