import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import os
import numpy as np
import time
import argparse

def easy_confusion(model_labels, caliop_labels, title="Model vs CALIOP", save=False, save_path=None):
    confusion_table = pd.crosstab(caliop_labels, model_labels, rownames=["Caliop"], colnames=["Model"], normalize="index")

    confusion_table.columns = ["Clear", "Water", "Ice"]
    confusion_table.index = ["Clear", "Water", "Ice"]

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot correlation matrix
    sns.heatmap(confusion_table.T, cmap="Blues", annot=True, fmt=".2f", linewidths=.5, ax=ax, cbar=False, vmin=0, vmax=1, annot_kws={"size": 14})

    # Set plot labels
    ax.set_title("Confusion Matrix")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    ax.xaxis.tick_top()  # Place x-axis labels on top
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("CALIOP", fontsize=12)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel("RF Model", fontsize=12)

    # Show the plot
    plt.show()

    if save:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    
    return confusion_table

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

# features_to_use = {
#     "infrared": [
#     'band_20', 'band_21', 'band_22', 'band_23', 'band_24', 'band_25', 'band_27', 'band_28', 'band_29', 'band_31', 'band_32', 'band_33', 'band_34', 'band_35', 'band_36',
#     'BT11', 'BT85', 'BT73', 'BTD85_11', 'BTD73_11', 'BT12',
#     #'mLong', 'mLat',
#     'mVZA', 'mSZA', #'day_of_year',
#     'mCloudTopTemp', 'mCloudTopPressure',
#     'mLST_night',
#     ],
#     "optical": [
#         'band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_7', 'band_17', 'band_18', 'band_19',
#         'band_20', 'band_21', 'band_22', 'band_23', 'band_24', 'band_25', 'band_26', 'band_27', 'band_28', 'band_29', 'band_31', 'band_32', 'band_33', 'band_34', 'band_35', 'band_36',
#         'BT11', 'BT85', 'BT73', 'BTD85_11', 'BTD73_11', 'BT12', 'ratio_R213_124',
#         #'mLong', 'mLat',
#         'mVZA', 'mSZA', #'day_of_year',
#         'mCloudTopTemp', 'mCloudTopPressure',
#         'mLST_day'
#     ]
# }

# second iteration
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

def main(args):
    folder = args.path

    if args.type == "infrared":
        clean_features = pd.read_csv(os.path.join(folder, "clean_infrared_features.csv"))
        clean_labels = pd.read_csv(os.path.join(folder, "clean_infrared_labels.csv"))
    else:
        clean_features = pd.read_csv(os.path.join(folder, "clean_optical_features.csv"))
        clean_labels = pd.read_csv(os.path.join(folder, "clean_optical_labels.csv"))

    train_data, test_data = test_train_split_day_batches(clean_features, test_size=0.15, random_seed=26)
    train_labels, test_labels = test_train_split_day_batches(clean_labels, test_size=0.15, random_seed=26)

    X = train_data[features_to_use[args.type]]
    y = train_labels["class"].values

    X_test = test_data[features_to_use[args.type]]
    y_test = test_labels["class"].values

    if args.type == "infrared":
        RF = RandomForestClassifier(max_depth=None, n_estimators=300, min_samples_leaf=5, max_features="sqrt", oob_score=True, n_jobs=args.njobs, random_state=26, verbose=1, ccp_alpha=0)
    else:
        RF = RandomForestClassifier(max_depth=None, n_estimators=500, min_samples_leaf=10, max_features="sqrt", oob_score=True, n_jobs=args.njobs, random_state=26, verbose=1, ccp_alpha=0)

    
    print(f"Starting RF {args.type} model training")
    RF.fit(X, y)
    print("Finished training")
    
    print("Starting permutation importance computation")
    t0 = time.time()
    result = permutation_importance(RF, X_test, y_test, n_repeats=10, random_state=42)
    t1 = time.time()
    computation_time = t1 - t0
    print("Finished permutation importance computation, time:", computation_time)
    print(f"Feature importances (name, mean, std): {zip(X_test.columns, result.importances_mean, result.importances_std)}")

    save_name = f"{args.type}_feature_importances{args.njobs}_jobs"

    # plot and save the feature importances
    plt.figure(figsize=(6, 6))
    sorted_idx = result.importances_mean.argsort()
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx])
    plt.title("Permutation Importances (test set)")
    plt.xlabel("Permutation Importance")
    plt.savefig(save_name + ".png")

    confusion_table = easy_confusion(RF.predict(X_test), y_test, title=args.type, save=False)

    # write this information to a log file
    with open(save_name + ".txt", "w") as f:
        if args.type == "infrared":
            f.write("Infrared model\n")
        if args.type == "optical":
            f.write("Optical model\n")
        f.write(f"Classifier parameters: {RF.get_params()}\n")
        f.write(f"OOB score: {RF.oob_score_:.4g}\n")
        f.write(f"Test accuracy: {RF.score(X_test, y_test):.4g}\n")
        f.write("Confusion matrix:\n")
        f.write(confusion_table.to_string())
        f.write("\n\n")
        f.write(f"Computation time: {int(computation_time//60)}min {int(computation_time%60)}s\n")
        f.write(f"Feature importances (name, mean, std): {zip(X_test.columns, result.importances_mean, result.importances_std)}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Feature Selection")
    parser.add_argument("--type", choices=["infrared", "optical"], help="The type of model to train")
    parser.add_argument("--path", type=str, help="Path to the data folder")
    parser.add_argument("--njobs", type=int, help="Number of threads to use")
    args = parser.parse_args()
    main(args)
