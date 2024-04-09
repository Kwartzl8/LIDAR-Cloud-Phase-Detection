from get_feature_sets import get_feature_sets, get_snow_ice_cover_classification
import argparse
import os
import pandas as pd

default_feature_sets = ["modis_infrared_cloud_phase", "modis_optical_cloud_phase", "modis_radiances", "modis_geo", "modis_cloud_top_properties", "snow_ice_classification", "surface_temperature", "caliop_cloud_phase"]

def main(args):
    # check if the collocation database exists
    if not os.path.exists(args.colpath):
        print("Collocation database not found. Exiting...")
        return
    
    # check if the feature set list and the data folder list are the same length
    if len(default_feature_sets) != len(args.datapath):
        print("Feature set list and data folder list must be the same length. Exiting...")
        return
    
    # put the paths in a dictionary so they can be accessed by the feature set name
    data_paths = dict(zip(default_feature_sets, args.datapath))

    # print the dictionary nicely
    print("Data paths:")
    for key, value in data_paths.items():
        # check if the path exists
        if os.path.exists(value):
            print(key, value, "ok")
        else:
            print(key, value, "not found")

    year_month_suffix = "_".join(args.colpath.split("/")[-3:-1])
    
    args.selected_features = dict(zip(default_feature_sets, args.selected_features))
    modis_features_keys = ["modis_radiances", "modis_geo", "snow_ice_classification", "surface_temperature", "modis_cloud_top_properties"]
    if any([args.selected_features.get(key) for key in modis_features_keys]):
        modis_training_features_list = []
        # get the modis_radiances, modis_geo (send in the pathname for snow_ice_classification to the optional argument), modis_cloud_top_properties and merge them together in a single dataframe

        print("Getting modis geo")
        modis_training_features_list.append(get_feature_sets("modis_geo", data_paths["modis_geo"], args.colpath, kwargs={"surface_datapath": data_paths["snow_ice_classification"], "myd11_path": data_paths["surface_temperature"]}))

        print("Getting modis cloud top properties")
        modis_training_features_list.append(get_feature_sets("modis_cloud_top_properties", data_paths["modis_cloud_top_properties"], args.colpath))

        print("Getting modis radiances")
        modis_training_features_list.append(get_feature_sets("modis_radiances", data_paths["modis_radiances"], args.colpath))
        
        modis_training_features = pd.merge(modis_training_features_list[0], modis_training_features_list[1], on=["fileID", "pixel_id"])
        # merge the dataframes matching filedID and pixel_id
        for i in range(2, len(modis_training_features_list)):
            modis_training_features = pd.merge(modis_training_features, modis_training_features_list[i], on=["fileID", "pixel_id"])

        modis_training_features = modis_training_features.rename(columns={"fileID": "mFileID"})

        # get rid of rows with duplicate indices
        modis_training_features = modis_training_features[~modis_training_features.index.duplicated(keep='first')]

        # the filename will be in the format modis_training_featuresYYYY_MM.csv and will be saved in the same folder as the collocation file
        modis_training_features_filename = "modis_training_features" + year_month_suffix + ".csv"
        # convert any boolean values to int, and limit the significant digits of floating point values to 4
        modis_training_features = modis_training_features.round(4)
        modis_training_features = modis_training_features.replace({True: 1, False: 0})
        modis_training_features.reset_index().to_csv(os.path.join(os.path.dirname(args.colpath), modis_training_features_filename), index=False)

    if args.selected_features["caliop_cloud_phase"]:
        # now get the caliop cloud phase and save it in a csv
        print("Getting caliop cloud phase")
        caliop_cloud_phase = get_feature_sets("caliop_cloud_phase", data_paths["caliop_cloud_phase"], args.colpath)
        caliop_cloud_phase_filename = "caliop_labels" + year_month_suffix + ".csv"
        caliop_cloud_phase.replace({True: 1, False: 0}).reset_index().to_csv(os.path.join(os.path.dirname(args.colpath), caliop_cloud_phase_filename), index=False)

    if args.selected_features["modis_infrared_cloud_phase"]:
        # get the ir modis cloud phase and save it in a csv
        print("Getting modis infrared cloud phase")
        modis_infrared_cloud_phase = get_feature_sets("modis_infrared_cloud_phase", data_paths["modis_infrared_cloud_phase"], args.colpath)
        # get rid of rows with duplicate indices
        modis_infrared_cloud_phase = modis_infrared_cloud_phase[~modis_infrared_cloud_phase.index.duplicated(keep='first')]
        modis_infrared_cloud_phase_filename = "modis_infrared_cloud_phase" + year_month_suffix + ".csv"
        modis_infrared_cloud_phase.replace({True: 1, False: 0}).reset_index().to_csv(os.path.join(os.path.dirname(args.colpath), modis_infrared_cloud_phase_filename), index=False)

    if args.selected_features["modis_optical_cloud_phase"]:
        # get the optical modis cloud phase and save it in a csv
        print("Getting modis optical cloud phase")
        modis_optical_cloud_phase = get_feature_sets("modis_optical_cloud_phase", data_paths["modis_optical_cloud_phase"], args.colpath)
        # get rid of rows with duplicate indices
        modis_optical_cloud_phase = modis_optical_cloud_phase[~modis_optical_cloud_phase.index.duplicated(keep='first')]
        modis_optical_cloud_phase_filename = "modis_optical_cloud_phase" + year_month_suffix + ".csv"
        modis_optical_cloud_phase.replace({True: 1, False: 0}).reset_index().to_csv(os.path.join(os.path.dirname(args.colpath), modis_optical_cloud_phase_filename), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for extracting all feature sets and glueing them together.")
    parser.add_argument("-colpath", type=str, help="Path to the collocation database.")

    default_feature_sets = ["modis_infrared_cloud_phase", "modis_optical_cloud_phase", "modis_radiances", "modis_geo", "modis_cloud_top_properties", "snow_ice_classification", "surface_temperature", "caliop_cloud_phase"]

    # data folder list for JASMIN
    default_data_folders = ["/neodc/modis/data/MYD06_L2/collection61",
                    "/neodc/modis/data/MYD06_L2/collection61",
                    "/neodc/modis/data/MYD021KM/collection61",
                    "/neodc/modis/data/MYD06_L2/collection61",
                    "/neodc/modis/data/MYD06_L2/collection61",
                    os.path.join(os.path.expanduser("~"), "MCD12C1"),
                    "/neodc/modis/data/MYD11A2/collection6",
                    "/gws/nopw/j04/gbov/data/asdc.larc.nasa.gov/data/CALIPSO/LID_L2_05kmMLay-Standard-V4-51/"]

    # give the option to specify the data folder
    parser.add_argument("-datapath", type=str, help="Paths to the data folders. Leave empty for the default data folder list to be used. Defaults: {}".format(default_data_folders), default=default_data_folders, nargs="+")

    # let the user select the feature sets to extract. The user will give a list of 0s and 1s to select the feature sets
    # the list will be in the same order as the default_feature_sets list
    # if the user gives a list of 0s and 1s, the default_feature_sets list will be filtered by the 1s
    # if the user gives an empty list, the default_feature_sets list will be used as is
    parser.add_argument("-selected_features", type=int, help="List of 0s and 1s to select the feature sets. Leave empty for the default feature set list to be used. Defaults: {}".format(default_feature_sets), default=[1] * len(default_feature_sets), nargs="+")

    # check if the selected feature list is the same length as the default feature list
    if len(default_feature_sets) != len(parser.parse_args().selected_features):
        print("Selected feature list and default feature list must be the same length. Exiting...")
        exit()

    args = parser.parse_args()
    main(args)