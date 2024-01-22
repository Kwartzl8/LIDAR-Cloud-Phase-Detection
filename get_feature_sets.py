from pyhdf.SD import *
import pandas as pd
import glob
import os
import numpy as np
import argparse

def get_caliop_id_from_filename(filename):
    return filename[-21:-2]

def get_caliop_cloud_phase(caliop_filepath):
    from caliop import Caliop_hdf_reader
    reader_caliop = Caliop_hdf_reader()
    caliop_df = pd.DataFrame(columns=["profile_id"])
    caliop_df.profile_id = reader_caliop._get_profile_id(caliop_filepath)
    caliop_df = caliop_df.set_index('profile_id')

    _, layer_type = reader_caliop._get_feature_classification(caliop_filepath, "Feature_Classification_Flags")
    caliop_df["cCloudy"] = np.where(np.any(layer_type == 2, axis=0), True, False)
    caliop_df["cClear"] = np.where(np.all(layer_type == 1, axis=0), True, False)
    caliop_df["cInvalid"] = np.where(np.any(layer_type == 0, axis=0), True, False)
    caliop_df["cAerosol"] = np.where((np.any(layer_type == 3, axis=0) |\
                                        np.any(layer_type == 4, axis=0)), True, False)

    cloud_phase_layer, cloud_phase_layer_qa = reader_caliop._get_cloud_phase(caliop_filepath, "Feature_Classification_Flags")

    caliop_df["cWater"] = np.where(np.any(cloud_phase_layer == 2, axis=0) &\
                                        ~np.any(cloud_phase_layer == 1, axis=0) &\
                                        ~np.any(cloud_phase_layer == 3, axis=0), True, False)
    caliop_df["cIce"] = np.where((np.any(cloud_phase_layer == 1, axis=0) |\
                                        np.any(cloud_phase_layer == 3, axis=0)) &\
                                        ~np.any(cloud_phase_layer == 2, axis=0), True, False)
    caliop_df["cUnknown"] = np.where(caliop_df["cCloudy"] &\
                                        np.any(cloud_phase_layer == 0, axis=0) &\
                                        ~np.any(cloud_phase_layer == 1, axis=0) &\
                                        ~np.any(cloud_phase_layer == 2, axis=0) &\
                                        ~np.any(cloud_phase_layer == 3, axis=0), True, False)
    caliop_df["cPhaseHighQA"] = np.where(np.any(cloud_phase_layer_qa == 3, axis=0), True, False)

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

def get_modis_5km_cloud_phase(modis_filepath, daytime_only=False):
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

    phase_downsampled = downsample_array_by_mode(phase_1km, pixelXsize=5, pixelYsize=5).flatten()

    df["pixel_id"] = np.arange(len(phase_downsampled))
    df.set_index("pixel_id", inplace=True)
    df["mClear"] = phase_downsampled == 0
    df["mWater"] = phase_downsampled == 1
    df["mIce"] = phase_downsampled == 2
    df["mUndetermined"] = phase_downsampled == 3

    if daytime_only:
        df["mValid"] = phase_downsampled != 4

    # should I set the cloudy flags according to the mask or according to the phase product?
    # cloud_mask_quality_flag = (MODIS_reader.select("Cloud_Mask_5km").get()[:,:,0] >> 1) & 0b11
    # df["mCloudy"] = (np.array(cloud_mask_quality_flag).flatten() == 0) | (np.array(cloud_mask_quality_flag).flatten() == 1)

    df["mCloudy"] = df["mWater"] | df["mIce"] | df["mUndetermined"]

    return df

def get_modis_5km_optical_cloud_phase(modis_filepath):
    return get_modis_5km_cloud_phase(modis_filepath, daytime_only=True)

def get_feature_sets(feature_set, data_folder, collocation_data_path):
    # define available functions and corresponding variables (TODO store in a text file)
    existing_feature_sets = ["modis_infrared_cloud_phase", "modis_optical_cloud_phase", "caliop_cloud_phase"]

    feature_set_extractor_func_dict = dict(zip(existing_feature_sets,\
                                    [get_modis_5km_cloud_phase, get_modis_5km_optical_cloud_phase, get_caliop_cloud_phase]))
    product_name_dict = dict(zip(existing_feature_sets, ["MYD06", "MYD06", "CAL_LID_L2_05kmMLay-Standard-V4-51"]))
    fileID_keys_dict = dict(zip(existing_feature_sets, ["mFileID", "mFileID", "cFileID"]))
    profileID_keys_dict = dict(zip(existing_feature_sets, ["pixel_id", "pixel_id", "profile_id"]))

    # parameter validation
    if feature_set not in existing_feature_sets:
        print(f"{feature_set} not in the existing feature set extractor functions. The parameter feature_set should be one of the following:")
        print(", ".join(existing_feature_sets))
        return

    if not os.path.exists(data_folder):
        print(f"The data folder given does not exist.")
        return
    
    # read in the collocation file
    collocation_df = pd.read_csv(collocation_data_path)
    collocated_file_ids = collocation_df[fileID_keys_dict[feature_set]]
    collocated_profile_ids = collocation_df[profileID_keys_dict[feature_set]]
    collocated_data_df_list = []

    # loop over unique file ids
    for file_id in set(collocated_file_ids):
        file_path = glob.glob(f"**/*{product_name_dict[feature_set]}*{file_id}*", root_dir=data_folder, recursive=True)

        if len(file_path) == 0:
            print(file_id, "file not found")
            raise Exception
        
        file_path = os.path.join(data_folder, file_path[0])
        
        # call the desired function
        df = feature_set_extractor_func_dict[feature_set](file_path)

        if df.empty:
            continue

        # reduce to collocated pixels
        collocated_profile_ids_in_file = collocated_profile_ids[collocated_file_ids == file_id]
        df = df.loc[collocated_profile_ids_in_file, :]

        # add the file id to uniquely identify each profile
        df["fileID"] = [file_id] * len(df)

        df.reset_index()
        df.set_index(["fileID"], append=True, inplace=True)

        collocated_data_df_list.append(df)

    return pd.concat(collocated_data_df_list)

def main(args):
    if args.s == "?":
        print("Give to the parameter -s the name of the file the feature set should be saved in.")
        return
    
    if os.path.exists(args.s):
        response = input(f"{args.s} already exists in this folder. Overwrite? (y/n)")
        if str.lower(response) != 'y':
            print("cancelled.")
            return

    df = get_feature_sets(args.f, args.datapath, args.colpath)
    df = df.astype(int)
    df.to_csv("./" + args.s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for extracting feature sets.")
    parser.add_argument("-f", type=str)
    parser.add_argument("-datapath", type=str)
    parser.add_argument("-colpath", type=str)
    parser.add_argument("-s", type=str, default="?")

    args = parser.parse_args()
    main(args)