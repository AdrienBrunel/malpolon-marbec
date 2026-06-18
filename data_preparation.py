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
splits = [0.70, 0.30]
rseed = 1312
train_ids, valid_ids = train_test_split(malpolon_df["survey_id"], test_size=splits[1], random_state=rseed)
malpolon_df.loc[malpolon_df["survey_id"].isin(train_ids), "subset"] = "train"
malpolon_df.loc[malpolon_df["survey_id"].isin(valid_ids), "subset"] = "val"
malpolon_df.to_csv(os.path.join(inp_dir, "malnutrition_dataset_malpolon.csv"), index=False)


# ==================================================== #
# PREPARE MALPOLON DATASET NPY SAMPLES
# ==================================================== #

# save samples individually
X_11x11 = np.load(os.path.join(raw_dir, "X_11x11.npy"), mmap_mode="r")
for k, survey_id in enumerate(malpolon_df["survey_id"]):
    np.save(os.path.join(inp_dir, "modality_11x11", "%s.npy" % survey_id), X_11x11[k])

X_21x21 = np.load(os.path.join(raw_dir, "X_21x21.npy"), mmap_mode="r")
for k, survey_id in enumerate(malpolon_df["survey_id"]):
    np.save(os.path.join(inp_dir, "modality_21x21", "%s.npy" % survey_id), X_21x21[k])

X_srtm = np.load(os.path.join(raw_dir, "X_srtm_3as.npy"), mmap_mode="r")
for k, survey_id in enumerate(malpolon_df["survey_id"]):
    np.save(os.path.join(inp_dir, "modality_srtm", "%s.npy" % survey_id), X_srtm[k])
