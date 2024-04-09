import os
import glob
import numpy as np
import pandas as pd
import argparse
import tqdm

def merge_features(folder_path, feature_name):
    # recursively find all files in this directory that contain the feature name in their filename
    feature_files = glob.glob("**/*" + feature_name + "*.csv", root_dir=folder_path, recursive=True)

    dataframes = []

    print("Merging " + feature_name + " features...")
    for file in tqdm.tqdm(feature_files):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)

def main(args):
    features_to_merge = ["merged_collocations", "caliop_labels", "modis_training_features", "modis_infrared_cloud_phase", "modis_optical_cloud_phase"]

    # reduce to the features that are selected
    features_to_merge = [feature for idx, feature in enumerate(features_to_merge) if args.selected_features[idx] != 0]
    # check if the collocation database exists
    if not os.path.isdir(args.path):
        raise ValueError("The provided path is not a valid directory.")
    
    # let the output folder be in the parent directory (a level higher) of the collocation database
    output_folder = os.path.join(args.path, "..", "merged_features")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for feature in features_to_merge:
        df = merge_features(args.path, feature)
        df.to_csv(os.path.join(output_folder, feature + ".csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for merging collocation csv files.")
    parser.add_argument("-path", type=str, help="path to collocation database folder", default="./collocation_database")
    parser.add_argument("-selected_features", type=int, nargs="+", help="list of selected features to merge {}".format(["merged_collocations", "caliop_labels", "modis_training_features", "modis_infrared_cloud_phase", "modis_optical_cloud_phase"]), default=[1, 1, 1, 1, 1, 1])

    args = parser.parse_args()
    main(args)