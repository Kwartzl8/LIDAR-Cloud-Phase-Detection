import os
import pandas as pd
import argparse

def collate_training_results(folder="./training_logs"):
    # get all the log files in the folder
    log_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    # create a dataframe to store the results
    results = pd.DataFrame(columns=["Model", "Feature subset", "Max depth", "Number of trees", "OOB score", "Computation time"])
    for file in log_files:
        with open(os.path.join(folder, file), "r") as f:
            # read the contents of the file
            content = f.readlines()
            # get the model type
            model = content[0].strip()
            # get the feature subset
            feature_subset = content[1].split(": ")[1].strip()
            # get the max depth
            max_depth = content[2].split(": ")[2].split(',')[0].strip()
            # get n_estimators
            n_estimators = content[2].split(": ")[3].split(',')[0].strip()
            # get the oob score
            oob_score = content[3].split(": ")[1].strip()
            # get the computation time
            computation_time = content[-1].split(": ")[1].strip()
            # concat this information to the dataframe
            results = pd.concat([results, pd.DataFrame({"Model": [model], "Feature subset": [feature_subset], "Max depth":[max_depth], "Number of trees": [n_estimators], "OOB score": [oob_score], "Computation time": [computation_time]})])
        
    # sort by the oob score
    results = results.sort_values(by="OOB score", ascending=False)
    # save the results to a csv file
    results.to_csv(os.path.join(folder, "training_results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get all training results together in a csv file")
    parser.add_argument("-folder", type=str, help="path to the folder containing the training logs", default="./training_logs")
    args = parser.parse_args()
    collate_training_results(args.folder)