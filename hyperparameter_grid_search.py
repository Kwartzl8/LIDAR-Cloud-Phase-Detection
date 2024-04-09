# usual imports
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import pandas as pd
import argparse
import warnings

from sklearn.ensemble import RandomForestClassifier
from itertools import product

def test_train_split_day_batches(data, test_size=0.2, random_seed=26):
    cycle_day = 20
    testing_days = int(test_size * cycle_day)

    if testing_days == 0:
        testing_days = 1

    datetimes = pd.to_datetime(data["cFileID"].apply(lambda x: x[0:10]), format="%Y-%m-%d")

    # get the earliest date in the dataset
    earliest_date = datetimes.min()
    days_since_earliest_date = (datetimes - earliest_date).dt.days

    # get the unique days list and shuffle them
    unique_days = days_since_earliest_date.unique()
    # np.random.seed(random_seed)
    # np.random.shuffle(unique_days)

    test_data = data[days_since_earliest_date.isin(unique_days[0::cycle_day])]
    for testing_day in range(1, testing_days):
        test_data = pd.concat([test_data, data[days_since_earliest_date.isin(unique_days[testing_day::cycle_day])]], axis=0)

    train_data = data[~data.index.isin(test_data.index)]

    return train_data, test_data

features_to_use = {
    "infrared": [
        'band_20', 'band_22', 'band_23', 'band_25', 'band_27', 'band_32', 'band_36',
        'BTD85_11', 'BTD73_11', 'BT73', 'BT12',
        'mLST_night',
        'mCloudTopTemp',
        'mVZA', 'mSZA'
    ],
    "optical": [
        'band_7', 'band_18', 'band_20', 'band_23', 'band_25', 'band_26', 'band_27', 'band_28', 'band_29', 'band_32', 'band_33',
        'ratio_R213_124',
        'mSZA',
        'mCloudTopPressure',
        'mLST_day'
    ]
}

tree_depths = [10, 15, 20, 40, 80, None]
# tree_depths = [20]
# n_estimators = [20, 50, 100, 150, 200, 250, 300, 400, 500]
n_estimators = [300, 500]
leaf_min_samples = [1, 2, 5, 10, 15, 20]
# leaf_min_samples = [1]

def main(args):
    folder = args.path
    
    # get time for the file name
    now = time.strftime("%H-%M")

    if args.type == "infrared":
        clean_features = pd.read_csv(os.path.join(folder, "clean_infrared_features.csv"))
        clean_labels = pd.read_csv(os.path.join(folder, "clean_infrared_labels.csv"))
    else:
        clean_features = pd.read_csv(os.path.join(folder, "clean_optical_features.csv"))
        clean_labels = pd.read_csv(os.path.join(folder, "clean_optical_labels.csv"))

    train_data, test_data = test_train_split_day_batches(clean_features, test_size=0.10, random_seed=26)
    train_labels, test_labels = test_train_split_day_batches(clean_labels, test_size=0.10, random_seed=26)

    train_data, val_data = test_train_split_day_batches(train_data, test_size=0.20)
    train_labels, val_labels = test_train_split_day_batches(train_labels, test_size=0.20)

    X_train = train_data[features_to_use[args.type]]
    y_train = train_labels["class"].values

    X_val = val_data[features_to_use[args.type]]
    y_val = val_labels["class"].values

    # X_test = test_data[features_to_use[args.type]]
    # y_test = test_labels["class"].values

    del train_data, train_labels, val_data, val_labels, test_data, test_labels

    # Ignore FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    with open(os.path.join(folder, f"hyperparameter_search{now}.txt"), "w") as f:
        f.write(f"Training with {args.type} data\n")
        f.write(f"Training data shape: {X_train.shape}\n")
        f.write(f"Validation data shape: {X_val.shape}\n\n")

    # create dataframe that will hold all possible hyperparameter combinations, and their scores
    logs_df = pd.DataFrame(columns=["tree_depth", "n_estimator", "leaf_min_sample", "train_score", "val_score"])

    # create iterator object for all possible hyperparameter combinations
    hyperparameter_iterator = product(tree_depths, n_estimators, leaf_min_samples)
    hyperparameter_combination_number = len(tree_depths) * len(n_estimators) * len(leaf_min_samples)

    logs_tree_depths = [0] * hyperparameter_combination_number
    logs_n_estimators = [0] * hyperparameter_combination_number
    logs_leaf_min_samples = [0] * hyperparameter_combination_number
    logs_train_scores = [0] * hyperparameter_combination_number
    logs_val_scores = [0] * hyperparameter_combination_number

    i = 0
    for tree_depth, n_estimator, leaf_min_sample in hyperparameter_iterator:
        print(f"Training with tree_depth: {tree_depth}, n_estimator: {n_estimator}, leaf_min_sample: {leaf_min_sample}")

        clf = RandomForestClassifier(
            n_estimators=n_estimator,
            max_depth=tree_depth,
            min_samples_leaf=leaf_min_sample,
            n_jobs=args.njobs,
            random_state=26
        )

        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        val_score = clf.score(X_val, y_val)

        print(f"Train score: {train_score}, Val score: {val_score}")

        with open(os.path.join(folder, f"hyperparameter_search{now}.txt"), "a") as f:
            f.write(f"Training with tree_depth: {tree_depth}, n_estimator: {n_estimator}, leaf_min_sample: {leaf_min_sample}\n")
            f.write(f"Train score: {train_score:.4g}, Val score: {val_score:.4g}\n")
        
        logs_tree_depths[i] = tree_depth
        logs_n_estimators[i] = n_estimator
        logs_leaf_min_samples[i] = leaf_min_sample
        logs_train_scores[i] = train_score
        logs_val_scores[i] = val_score

        i += 1

    # save in dataframe
    logs_df["tree_depth"] = logs_tree_depths
    logs_df["n_estimator"] = logs_n_estimators
    logs_df["leaf_min_sample"] = logs_leaf_min_samples
    logs_df["train_score"] = logs_train_scores
    logs_df["val_score"] = logs_val_scores

    # save logs as csv
    logs_df.to_csv(os.path.join(folder, f"hyperparameter_search{now}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Feature Selection")
    parser.add_argument("--type", choices=["infrared", "optical"], help="The type of model to train")
    parser.add_argument("--path", type=str, help="Path to the data folder")
    parser.add_argument("--njobs", type=int, help="Number of threads to use")
    args = parser.parse_args()
    main(args)