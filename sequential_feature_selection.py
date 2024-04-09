import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.base import clone
from numbers import Integral, Real
import warnings
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
    # plt.show()

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

features_to_use = {
    "infrared": [
    'band_20', 'band_21', 'band_22', 'band_23', 'band_24', 'band_25', 'band_27', 'band_28', 'band_29', 'band_31', 'band_32', 'band_33', 'band_34', 'band_35', 'band_36',
    'BT11', 'BT85', 'BT73', 'BTD85_11', 'BTD73_11', 'BT12',
    #'mLong', 'mLat',
    'mVZA', 'mSZA', #'day_of_year',
    'mCloudTopTemp', 'mCloudTopPressure',
    'mLST_night',
    ],
    "optical": [
        'band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_7', 'band_17', 'band_18', 'band_19',
        'band_20', 'band_21', 'band_22', 'band_23', 'band_24', 'band_25', 'band_26', 'band_27', 'band_28', 'band_29', 'band_31', 'band_32', 'band_33', 'band_34', 'band_35', 'band_36',
        'BT11', 'BT85', 'BT73', 'BTD85_11', 'BTD73_11', 'BT12', 'ratio_R213_124',
        #'mLong', 'mLat',
        'mVZA', 'mSZA', #'day_of_year',
        'mCloudTopTemp', #'mCloudTopPressure',
        'mLST_day'
    ]
}

# second iteration
features_to_use = {
    "infrared": [
        'band_20', 'band_22', 'band_23', 'band_25', 'band_27', 'band_32', 'band_36',
        'BTD85_11', 'BTD73_11', 'BT73', 'BT12', 'BTD12_11', 'BT67', 'BTD39_11', 'BTD39_12',
        'mLST_night',
        'mCloudTopTemp',
        'mVZA', 'mSZA'
    ],
    "optical": [
        'band_1', 'band_3', 'band_4', 'band_5', 'band_17', 'band_19',
        'band_7', 'band_18',
        'band_20', 'band_25', 'band_26', 'band_27', 'band_28', 'band_29',
        'mSZA', 'mCloudTopPressure',
        'BTD85_11', 'BTD73_11', 'BT73', 'BT12', 'BTD12_11', #'BT67', 'BTD39_11', 'BTD39_12',
    ]
}

initial_features = {
    "infrared": [
        'band_20', 'band_22', 'band_23', 'band_25', 'band_27', 'band_32', 'band_36',
        'BTD85_11', 'BTD73_11', 'BT73', 'BT12', 'BTD12_11', 'BT67', 'BTD39_11', 'BTD39_12',
        'mLST_night',
        'mCloudTopTemp',
        'mVZA', 'mSZA'
    ],
    "optical": [
        'band_1', 'band_3', 'band_4', 'band_5', 'band_17', 'band_19',
        'band_7', 'band_18',
        'band_20', 'band_25', 'band_26', 'band_27', 'band_28', 'band_29',
        'mSZA', 'mCloudTopPressure',
        'BTD85_11', 'BTD73_11', 'BT73', 'BT12', 'BTD12_11', #'BT67', 'BTD39_11', 'BTD39_12',
    ]
}

n_features_for_model = {
    "infrared": 10,
    "optical": 10
}

class ValidationSequentialFeatureSelector(SequentialFeatureSelector):
    def __init__(self, estimator, validation_data, validation_labels, initial_feature_mask, n_features_to_select='auto', direction='forward', n_jobs=None, verbose=0):
        super().__init__(estimator, n_features_to_select=n_features_to_select, direction=direction, n_jobs=n_jobs)
        self.validation_data = validation_data
        self.validation_labels = validation_labels
        self.verbose = verbose
        self.initial_feature_mask = initial_feature_mask

    def fit(self, X, y=None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()

        # FIXME: to be removed in 1.3
        if self.n_features_to_select in ("warn", None):
            # for backwards compatibility
            warnings.warn(
                "Leaving `n_features_to_select` to "
                "None is deprecated in 1.0 and will become 'auto' "
                "in 1.3. To keep the same behaviour as with None "
                "(i.e. select half of the features) and avoid "
                "this warning, you should manually set "
                "`n_features_to_select='auto'` and set tol=None "
                "when creating an instance.",
                FutureWarning,
            )

        tags = self._get_tags()
        X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )
        n_features = X.shape[1]
        n_initial_features = sum(self.initial_feature_mask)

        # FIXME: to be fixed in 1.3
        error_msg = (
            "n_features_to_select must be either 'auto', 'warn', "
            "None, an integer in [1, n_features - 1] "
            "representing the absolute "
            "number of features, or a float in (0, 1] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )
        if self.n_features_to_select in ("warn", None):
            if self.tol is not None:
                raise ValueError("tol is only enabled if `n_features_to_select='auto'`")
            self.n_features_to_select_ = n_features // 2
        elif self.n_features_to_select == "auto":
            if self.tol is not None:
                # With auto feature selection, `n_features_to_select_` will be updated
                # to `support_.sum()` after features are selected.
                self.n_features_to_select_ = n_features - 1
            else:
                self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):
            if not 0 < self.n_features_to_select < n_features:
                raise ValueError(error_msg)
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, Real):
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)

        if self.tol is not None and self.tol < 0 and self.direction == "forward":
            raise ValueError("tol must be positive when doing forward selection")

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        if self.direction == 'forward':
            current_mask = self.initial_feature_mask
        else:
            current_mask = ~self.initial_feature_mask
        
        n_iterations = (
            self.n_features_to_select_ - n_initial_features
            if self.n_features_to_select == "auto" or self.direction == "forward"
            else n_initial_features - self.n_features_to_select_
        )

        old_score = -np.inf
        is_auto_select = self.tol is not None and self.n_features_to_select == "auto"
        for _ in range(n_iterations):
            new_feature_idx, new_score = self._get_best_new_feature_score(
                cloned_estimator, X, y, current_mask
            )
            if is_auto_select and ((new_score - old_score) < self.tol):
                break

            old_score = new_score
            current_mask[new_feature_idx] = True

        if self.direction == "backward":
            current_mask = ~current_mask

        self.support_ = current_mask
        self.n_features_to_select_ = self.support_.sum()

        return self

    def _get_best_new_feature_score(self, estimator, X, y, current_mask):
        # Return the best new feature and its score to add to the current_mask,
        # i.e. return the best new feature and its score to add (resp. remove)
        # when doing forward selection (resp. backward selection).
        # Feature will be added if the current score and past score are greater
        # than tol when n_feature is auto,
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}

        print(f"Currently considering features {self.validation_data.columns[current_mask].values}")

        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == "backward":
                candidate_mask = ~candidate_mask

            X_train = X[:, candidate_mask]
            estimator.fit(X_train, y)
            scores[feature_idx] = estimator.score(self.validation_data.loc[:, candidate_mask].values, self.validation_labels)
            print(f"Score for added feature {self.validation_data.columns[feature_idx]} is {scores[feature_idx]:.4g}") if self.verbose else None

        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
        print(f"FEATURE {self.validation_data.columns[new_feature_idx]} SELECTED, WITH SCORE {scores[new_feature_idx]:.4g}")
        return new_feature_idx, scores[new_feature_idx]

def main(args):
    folder = args.path
    number_of_features_to_select = n_features_for_model[args.type]

    if args.type == "infrared":
        clean_features = pd.read_csv(os.path.join(folder, "clean_infrared_features.csv"))
        clean_labels = pd.read_csv(os.path.join(folder, "clean_infrared_labels.csv"))
    else:
        clean_features = pd.read_csv(os.path.join(folder, "clean_optical_features.csv"))
        clean_labels = pd.read_csv(os.path.join(folder, "clean_optical_labels.csv"))

    initial_features_mask = np.isin(features_to_use[args.type], initial_features[args.type])

    train_data, test_data = test_train_split_day_batches(clean_features, test_size=0.10, random_seed=26)
    train_labels, test_labels = test_train_split_day_batches(clean_labels, test_size=0.10, random_seed=26)

    train_data, val_data = test_train_split_day_batches(train_data, test_size=0.20)
    train_labels, val_labels = test_train_split_day_batches(train_labels, test_size=0.20)

    X_train = train_data[features_to_use[args.type]]
    y_train = train_labels["class"].values

    X_val = val_data[features_to_use[args.type]]
    y_val = val_labels["class"].values

    X_test = test_data[features_to_use[args.type]]
    y_test = test_labels["class"].values


    if args.type == "infrared":
        RF = RandomForestClassifier(max_depth=40, n_estimators=300, min_samples_leaf=2, max_features="sqrt", oob_score=True, n_jobs=args.njobs, random_state=26, verbose=0, ccp_alpha=0)
    else:
        RF = RandomForestClassifier(max_depth=None, n_estimators=300, min_samples_leaf=2, max_features="sqrt", oob_score=True, n_jobs=args.njobs, random_state=26, verbose=0, ccp_alpha=0)

    sfs = ValidationSequentialFeatureSelector(RF, X_val, y_val, initial_features_mask, n_features_to_select=number_of_features_to_select, direction="backward", verbose=1)

    print(f"Starting {sfs.get_params()['direction']} selection sequence")
    sfs.fit(X_train, y_train)
    print("Finished selection sequence")

    print("Selected features:", sfs.get_support(indices=True))
    selected_features = X_train.columns[sfs.get_support()]
    print("Selected features:", selected_features)

    print("Fitting model with selected features")
    t0 = time.time()
    RF.fit(X_train[selected_features], y_train)
    t1 = time.time()
    computation_time = t1 - t0

    model_labels = RF.predict(X_test[selected_features])

    save_name = f"{args.type}_{number_of_features_to_select}_features_{sfs.get_params()['direction']}_{args.njobs}_jobs"

    confusion_table = easy_confusion(model_labels, y_test, title="Sequential Feature Selection", save=True, save_path=save_name + ".png")

    # write this information to a log file
    with open(save_name + ".txt", "w") as f:
        if args.type == "infrared":
            f.write("Infrared model\n")
            f.write(f"Features selected: {selected_features}\n")
        if args.type == "optical":
            f.write("Optical model\n")
            f.write(f"Feature selected: {selected_features}\n")
        f.write(f"Classifier parameters: {RF.get_params()}\n")
        f.write(f"OOB score: {RF.oob_score_:.4g}\n")
        f.write(f"Test accuracy: {RF.score(X_test[selected_features], y_test):.4g}\n")
        f.write("Confusion matrix:\n")
        f.write(confusion_table.to_string())
        f.write("\n\n")
        f.write(f"Computation time: {int(computation_time//60)}min {int(computation_time%60)}s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Feature Selection")
    parser.add_argument("--type", choices=["infrared", "optical"], help="The type of model to train")
    parser.add_argument("--path", type=str, help="Path to the data folder")
    parser.add_argument("--njobs", type=int, help="Number of threads to use")
    args = parser.parse_args()
    main(args)
