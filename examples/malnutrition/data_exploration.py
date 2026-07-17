# ==================================================== #
# LIBRARIES
# ==================================================== #
import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from examples.malnutrition.src import utils


# ==================================================== #
# PARAMETERS
# ==================================================== #
modalities = ["mod_bb20km_123y", "mod_bb50km_123y", "mod_bb100km_123y"]#, "mod_images"]
data_names = ["bb20km", "bb50km", "bb100km"]#, "images"]
study_name = "malnutrition_training_mod_bb20_bb50_bb100_img_123y_nbchildren10-2026-07-16_15-45"


# ==================================================== #
# DIRECTORIES
# ==================================================== #
root_dir = os.getcwd()
inp_dir = os.path.join("/marbec-data","Fish4Nutrition", "malpolon-marbec", "examples", "malnutrition", "data", "inputs")
raw_dir = os.path.join("/marbec-data","Fish4Nutrition", "malpolon-marbec", "examples", "malnutrition", "data", "raw", "v2")
dhs_dir = os.path.join("/marbec-data","Fish4Nutrition", "malpolon-marbec", "examples", "malnutrition", "data", "raw")
plot_dir = os.path.join(root_dir, "examples", "malnutrition", "outputs", study_name, "inputs")


# ==================================================== #
# LOAD DATA
# ==================================================== #
# load dhs data
dhs_data = pd.read_csv(os.path.join(dhs_dir, "dhs_malnutrition_prevalence_by_cluster.csv"))
n_dhs_data = len(dhs_data)

# load malpolon data
suffix = "nbchildren10"
malpolon_df = pd.read_csv(os.path.join(inp_dir, "malnutrition_dataset_malpolon_%s.csv") % suffix)


# ==================================================== #
# PLOTS
# ==================================================== #
# set samples to be plotted
samples = np.random.randint(len(malpolon_df), size=10)

# loop over samples
for sample in samples:

    # loop over modalities
    for modality, data_name in zip(modalities, data_names):

        # load numpy dataset
        survey_id = malpolon_df.loc[sample,"survey_id"]
        X_features = np.lib.format.open_memmap(os.path.join(inp_dir, modality, "%s.npy" % survey_id), mode="r+")
        with open(os.path.join(raw_dir, "metadata_%s.pkl" % data_name), "rb") as f:
            metadata = pickle.load(f)
        metadata_df = pd.DataFrame([])
        for key in metadata.keys():
            metadata_df = pd.concat((metadata_df, metadata[key]["df"])).reset_index(drop=True)

        # compute layout for plotting
        n_plots = X_features.shape[0]
        n_cols, n_rows = utils.nearsq_grid_layout(n_plots)
        
        # plot data
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(40, 40))
        if n_plots > 1:
            v = 0
            axes = axes.flatten()
            for ax in axes:
                if v < n_plots:
                    _ = ax.imshow(X_features[v], cmap="viridis")
                    _ = ax.set_title(metadata_df.loc[v, "feature_name"], fontsize=25)
                    _ = ax.axis("off")
                    v = v+1
                else:
                    _ = ax.axis("off")
            _ = plt.tight_layout(pad=1.0, w_pad=1.5, h_pad=2.0)
        else:
            _ = plt.imshow(X_features, cmap="viridis")
            _ = plt.title("[%d] - %s" % (sample, data_name), fontsize=25)
            _ = plt.axis("off")

        # save figure
        plt.savefig(os.path.join(plot_dir, "sample_%d_%s.png" % (sample, data_name)), dpi=50)
        fig.clear()
        plt.close(fig)