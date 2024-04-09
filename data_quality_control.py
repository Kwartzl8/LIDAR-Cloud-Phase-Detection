import numpy as np
import pandas as pd
import os

def caliop_cleanup(df, verbose):
    # delete rows with cInvalid == 1 or cAerosolFree == 0
    df = df[(df.cInvalid == 0) & (df.cAerosolFree == 1)].drop(columns=["cInvalid", "cAerosolFree"])
    print("Only keep valid and aerosol free profiles", df.shape) if verbose else None

    # not needed anymore, got rid of these since I will not be using them anyway
    # # drop columns cCloudy, cWater, cIce, cMixedMultilayer, cUnknown
    # df = df.drop(columns=["cCloudy", "cWater", "cIce", "cMixedMultilayer", "cUnknown"])

    # rename cCloudyHighQA to cCloudy, cWaterDominant to cWater, cIceDominant to cIce, cMixedMultilayerHighQA to cMixedMultilayer
    df.rename(columns={"cCloudyHighQA": "cCloudy", "cWaterDominant": "cWater", "cIceDominant": "cIce", "cMixedMultilayerHighQA": "cMixedMultilayer", "cUnknownHighQA": "cUnknown"}, inplace=True)

    # are there data points with both cCloudy and cClear set to 0?
    df = df[(df.cCloudy == 1) | (df.cClear == 1)]
    print("Eliminated profiles with weak cloud evidence", df.shape) if verbose else None

    # delete rows with cUnknown == 1
    df = df[((df.cUnknown == 0) & (df.cCloudy == 1)) | (df.cClear == 1)].drop(columns=["cUnknown"])
    print("Only keep known phase classifications", df.shape) if verbose else None

    # because I have reclassified cMixedMultilayer to cWater and cIce, remove the rows with cAmbiguous == 1 and remove the column
    df = df[df.cAmbiguous == 0].drop(columns=["cAmbiguous"])
    print("Only keep profiles with no ambiguous phase", df.shape) if verbose else None

    # check whether the c columns are mutually exclusive
    mutually_exclusive = ((df.cWater == 1) & (df.cIce == 1)).any() | \
                            ((df.cClear == 1) & (df.cWater == 1)).any() | \
                            ((df.cIce == 1) & (df.cClear == 1)).any()

    if mutually_exclusive:
        print("Warning: some profiles have multiple phase classifications")

    # for now, drop all columns that are not needed
    df = df.drop(columns=["cMixedMultilayer", "cIceHighQA", "cWaterHighQA", "cIAB_ice", "cIAB_water"])

    return df

def data_quality_control(folder, verbose=False):
    # check whether the folder exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist")

    # check if the folder ends with a slash. if not, add it
    if folder[-1] != "/":
        folder += "/"

    collocation_df = pd.read_csv(folder + "merged_collocations.csv")
    modis_training_features = pd.read_csv(folder + "modis_training_features.csv")
    caliop_labels = pd.read_csv(folder + "caliop_labels.csv")
    modis_infrared_phase = pd.read_csv(folder + "modis_infrared_cloud_phase.csv")
    modis_optical_phase = pd.read_csv(folder + "modis_optical_cloud_phase.csv")

    if verbose:
        print("Collocation database", collocation_df.shape)
        print(collocation_df.head())

        print("MODIS training features", modis_training_features.shape)
        print(modis_training_features.head())

        print("CALIOP labels", caliop_labels.shape)
        print(caliop_labels.head())

        print("MODIS infrared phase", modis_infrared_phase.shape)
        print(modis_infrared_phase.head())

        print("MODIS optical phase", modis_optical_phase.shape)
        print(modis_optical_phase.head())
    
    # merge collocation with caliop labels
    all_features_with_labels = pd.merge(collocation_df, caliop_labels.rename(columns={"fileID": "cFileID"}), on=["cFileID", "profile_id"], how="inner")
    # merge this with the training features
    all_features_with_labels = pd.merge(all_features_with_labels, modis_training_features.rename(columns={"fileID": "mFileID"}), on=["mFileID", "pixel_id"], how="inner")

    algorithms = ["infrared", "optical"]
    for algorithm in algorithms:
        # merge df with the phase as classified by the chosen algorithm
        if algorithm == "infrared":
            df = pd.merge(all_features_with_labels, modis_infrared_phase.rename(columns={"fileID": "mFileID"}), on=["mFileID", "pixel_id"], how="inner")
        elif algorithm == "optical":
            df = pd.merge(all_features_with_labels, modis_optical_phase.rename(columns={"fileID": "mFileID"}), on=["mFileID", "pixel_id"], how="inner")

        # remove all rows that have duplicates in the pair (pixel_id, mFileID)
        df.drop_duplicates(subset=["pixel_id", "mFileID"], inplace=True)
        print(f"All features with MODIS phase and CALIOP labels", df.shape) if verbose else None

        # delete rows with mSnowIceCover == 0
        df = df[df.mSnowIceCover == 1].drop(columns=["mSnowIceCover"])
        print("Only datapoints above ice and snow", df.shape) if verbose else None

        # drop beta columns as there are too many missing values
        df = df.drop(columns=[col for col in df.columns if "Beta" in col])

        df = caliop_cleanup(df, verbose)

        df = df[~(df.mCloudTopPressure.isna() & df.mCloudy == 1)]
        print("Only keep cloudy (MODIS classified) datapoints with valid mCloudTopPressure and mCloudTopTemp", df.shape) if verbose else None

        # replace NaN values in the mCloudTopPressure with -100, as the NaN values left are likely to be clear sky
        df.mCloudTopPressure.fillna(-100, inplace=True)
        # replace NaN values in the mCloudTopTemp with -100
        df.mCloudTopTemp.fillna(-100, inplace=True)

        # prepare the dataframe for saving for the algorithm chosen
        if algorithm == "infrared":
            infrared_band_list = [f"band_{i}" for i in range(20, 37)]

            # remove band_26 (SWIR, only available during daylight) and band_30 (ozone band, should not be taken into account) from list
            infrared_band_list.remove("band_26")
            # infrared_band_list.remove("band_30")

            # remove all bands from the dataframe that are not in the infrared_band_list
            df = df.drop(columns=[col for col in df.columns if col not in infrared_band_list and "band" in col])
            print("Dropped all daylight bands", df.shape) if verbose else None

            # remove all rows with mSZA < 81.36 because I want this to be the nighttime only data
            df = df[df.mSZA > 81.36]
            print("Only keep nighttime pixels", df.shape) if verbose else None

            # no need to do this, as I have removed the daylight pixels
            # # replace all mLST_day values with -100 if mSZA > 81.36 and all mLST_night values with -100 if mSZA <= 81.36
            # df.loc[df.mSZA > 81.36, "mLST_day"] = np.full(df[df.mSZA > 81.36].index.values.shape, -100)
            # df.loc[df.mSZA <= 81.36, "mLST_night"] = np.full(df[df.mSZA <= 81.36].index.values.shape, -100)

            # drop mLST_day column
            df = df.drop(columns=["mLST_day"])

            # # drop all rows that have surface temperature nan values in both day and night columns
            # df = df[~((df["mLST_night"] != -100) & (df["mLST_day"] != -100))]
            # print("Dropped all rows with LST values in both day and nighttime", df.shape) if verbose else None

        elif algorithm == "optical":
            # drop all rows in df with mOptical == 0
            df = df[df.mOptical == 1].drop(columns=["mOptical"])
            print("Only keep profiles that have been classified by MODIS with the optical algorithm", df.shape) if verbose else None

            # drop bands 8-16, as they are for ocean stuff and seem to also produce a lot of NaN values
            ocean_bands_to_remove = [f"band_{i}" for i in range(8, 17)]
            # bands 13 and 14 have different names
            ocean_bands_to_remove.remove("band_13")
            ocean_bands_to_remove.remove("band_14")
            # add back 4 bands, 13lo, 13hi, 14lo, 14hi
            ocean_bands_to_remove.extend(["band_13lo", "band_13hi", "band_14lo", "band_14hi"])

            df = df.drop(columns=ocean_bands_to_remove)

            # also drop LST_night
            df = df.drop(columns=["mLST_night"])

        # remove all rows with NaN values
        df = df.dropna()
        print("Drop all nan values left", df.shape) if verbose else None

        df.drop(columns=["cCloudy", "cClear", "cWater", "cIce"]).to_csv(folder + f"clean_{algorithm}_features.csv", index=False)
        print(f"Saved cleaned features for {algorithm} algorithm") if verbose else None

        labels = df[["profile_id", "cFileID", "cCloudy", "cClear", "cWater", "cIce", "cCloudTopHeight", "cCloudGeometricalThickness", "cColumnOpticalDepth"]].set_index(["profile_id", "cFileID"])
        labels['class'] = np.where(labels.cWater == 1, 1, 0) + np.where(labels.cIce == 1, 2, 0)

        # labelsY = pd.DataFrame(labelsY, columns=["class"], index=labels.index)

        labels.to_csv(folder + f"clean_{algorithm}_labels.csv")