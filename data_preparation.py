# ==================================================== #
# LIBRARIES
# ==================================================== #
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================================== #
# DIRECTORIES
# ==================================================== #
root_dir = os.getcwd()
raw_dir = os.path.join(root_dir, "examples", "malnutrition", "data", "raw")
inp_dir = os.path.join(root_dir, "examples", "malnutrition", "data", "inputs")


# ==================================================== #
# LOAD DATA
# ==================================================== #
dhs_data = pd.read_csv(os.path.join(raw_dir, "dhs_malnutrition_prevalence_by_cluster.csv"))
n_dhs_data = len(dhs_data)


# ==================================================== #
# PREPARE MALPOLON DATASET CSV
# ==================================================== #

# build dataframe for malpolon dataset
malpolon_df = pd.DataFrame([], columns=["survey_id", "eventDate", "y_ms_stunting", "subset"])
malpolon_df["survey_id"] = dhs_data["id"]
malpolon_df["y_ms_stunting"] = dhs_data["moderate_and_severe_stunting"]

# split dataset
splits = [0.70, 0.15, 0.15]
rseed_split = 1312
train_ids, valid_ids = train_test_split(malpolon_df["survey_id"], test_size=splits[1], random_state=rseed_split)
train_ids, test_ids = train_test_split(train_ids, test_size=splits[2]/(1-splits[1]), random_state=rseed_split)
n_samples, n_train_samples, n_valid_samples, n_test_samples = len(malpolon_df), len(train_ids), len(valid_ids), len(test_ids)
print("samples [%d] = train [%d | %.1f%%] + validation [%d | %.1f%%] + test [%d | %.1f%%]" % (n_samples, n_train_samples, 100*n_train_samples/n_samples, n_valid_samples, 100*n_valid_samples/n_samples, n_test_samples, 100*n_test_samples/n_samples))

malpolon_df.loc[malpolon_df["survey_id"].isin(train_ids), "subset"] = "train"
malpolon_df.loc[malpolon_df["survey_id"].isin(valid_ids), "subset"] = "val"
malpolon_df.loc[malpolon_df["survey_id"].isin(test_ids), "subset"] = "test"
malpolon_df.to_csv(os.path.join(inp_dir, "malnutrition_dataset_malpolon.csv"), index=False)


# ==================================================== #
# PREPARE MALPOLON DATASET
# ==================================================== #
# load data
malpolon_df = pd.read_csv(os.path.join(inp_dir, "malnutrition_dataset_malpolon.csv"))
X_11x11 = np.load(os.path.join(raw_dir, "X_11x11.npy"))
X_21x21 = np.load(os.path.join(raw_dir, "X_21x21.npy"))
X_srtm = np.load(os.path.join(raw_dir, "X_srtm_3as.npy"))

print(X_11x11.shape)
print(X_11x11.dtype)
print(X_21x21.shape)
print(X_21x21.dtype)
print(X_srtm.shape)
print(X_srtm.dtype)

# replace nans with 0.0
X_11x11[np.isnan(X_11x11)] = 0.0
X_21x21[np.isnan(X_21x21)] = 0.0
X_srtm[np.isnan(X_srtm)] = 0

# remove channels that are all zeros
keep_channels = np.any(X_21x21 != 0, axis=(0, 2, 3))
X_21x21 = X_21x21[:, keep_channels, :, :]

# standardize data
channel_mean = np.nanmean(X_11x11, axis=(0, 2, 3), keepdims=True)
channel_std = np.nanstd(X_11x11, axis=(0, 2, 3), keepdims=True)
X_11x11_standardized = (X_11x11-channel_mean)/channel_std

channel_mean = np.nanmean(X_21x21[:,0:6,:,:], axis=(0, 2, 3), keepdims=True)
channel_std = np.nanstd(X_21x21[:,0:6,:,:], axis=(0, 2, 3), keepdims=True)
X_21x21_standardized = np.concatenate([((X_21x21[:,0:6,:,:]-channel_mean)/channel_std), X_21x21[:,6:,:,:]], axis=1)

channel_mean = np.nanmean(X_srtm, axis=(0, 2, 3), keepdims=True)
channel_std = np.nanstd(X_srtm, axis=(0, 2, 3), keepdims=True)
X_srtm_standardized = (X_srtm - channel_mean)/channel_std

# save samples individually
for k, survey_id in enumerate(malpolon_df["survey_id"]):
    np.save(os.path.join(inp_dir, "modality_11x11", "%s.npy" % survey_id), X_11x11[k])
    np.save(os.path.join(inp_dir, "modality_21x21", "%s.npy" % survey_id), X_21x21[k])
    np.save(os.path.join(inp_dir, "modality_srtm", "%s.npy" % survey_id), X_srtm[k])
