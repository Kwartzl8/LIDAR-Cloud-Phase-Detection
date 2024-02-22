import os
import glob
import numpy as np
import pandas as pd
import argparse
import tqdm

def merge_collocation_data(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError("The provided path is not a valid directory.")

    # recursively find all files in this directory that match the pattern \-dd\Tdd\- (to make sure we are not reading other csvs)
    collocation_files = glob.glob("**/*-[0-9][0-9]T[0-9][0-9]-*.csv", root_dir=folder_path, recursive=True)

    def get_caliop_id_from_filename(filename):
        return filename[-21:-2]

    dataframes = []

    for file in tqdm.tqdm(collocation_files):
        file_path = os.path.join(folder_path, file)
        with open(file_path) as f:
            start_and_end_times = f.readline()
            # make a dictionary so the file number can be replaced with the string
            modis_search_strings = {i: file for (i, file) in enumerate(f.readline().split(' ')[:-1])}
        
        df = pd.read_csv(file_path, header=2).rename(columns={"MODIS_file": "mFileID", "modis_idx": "pixel_id"})
        df.mFileID = np.vectorize(modis_search_strings.get)(df.mFileID.values)
        caliop_id = get_caliop_id_from_filename(os.path.basename(file_path)[:-4])
        df["cFileID"] = [caliop_id] * len(df.profile_id.values)

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)

def main(args):
    df = merge_collocation_data(args.path)

    # check if the output path is provided
    if args.o is None:
        filename = "merged_collocations"
        # add the names of the last 2 folders to the filename separated by underscores
        filename += "_".join(args.path.split("/")[-2:]) + ".csv"
        
        args.o = os.path.join(args.path, filename)
    
    df.to_csv(args.o, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for merging collocation csv files.")
    parser.add_argument("-path", type=str, help="path to collocation database folder", default="./collocation_database")
    parser.add_argument("-o", type=str, help="output file path", required=False)

    args = parser.parse_args()
    main(args)