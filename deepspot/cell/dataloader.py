from deepspot.utils.utils_dataloader import spatial_upsample_and_smooth
from deepspot.utils.utils_dataloader import add_zero_padding
from deepspot.utils.utils_dataloader import load_data
from deepspot.utils.utils_dataloader import get_balanced_index

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from distutils.dir_util import copy_tree
from torch.utils.data import Dataset
from collections import Counter

from tqdm import tqdm
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import torch
from scipy.stats import rankdata


class DeepCellDataLoader(Dataset):
    def __init__(self,
                 out_folder,
                 samples,
                 genes_to_keep,
                 morphology_model_name,
                 cell_diameter,
                 target_sum=10000,
                 resolution=0,
                 cell_context='cell_neighbors',
                 radius_neighbors=0,
                 augmentation="default",
                 normalize=None,
                 scaler=None,
                 factor_log1p=10000,
                 resample_samples=False,
                 smooth_n=0):
        super().__init__()
        self.out_folder = out_folder
        self.samples = samples
        self.cell_diameter = cell_diameter
        self.cell_context = cell_context
        self.genes_to_keep = genes_to_keep
        self.target_sum = target_sum
        self.morphology_model_name = morphology_model_name
        self.resolution = resolution
        self.radius_neighbors = radius_neighbors
        self.max_n_neighbors = 0
        self.augmentation = augmentation
        self.smooth_n = smooth_n
        self.normalize = normalize
        self.scaler = scaler
        self.factor_log1p = factor_log1p
        self.resample_samples = resample_samples
        data = load_data(samples, self.out_folder,
                         self.morphology_model_name,
                         load_image_features=False,
                         factor=self.factor_log1p)

        y = data["y"][:, self.genes_to_keep]

        transcriptomics = pd.DataFrame(y, index=data["barcode"])

        coordinates_df = []
        for sample in tqdm(samples):

            adata_path = f"{out_folder}/data/h5ad/{sample}.h5ad"
            adata_obs = sc.read_h5ad(adata_path)

            sample_idx = transcriptomics.index.to_series().apply(lambda x: x.split("_")[1] == sample)
            counts = transcriptomics[sample_idx]
            barcode = counts.index.to_series().apply(lambda x: x.split("_")[0])
            adata_train = ad.AnnData(counts.values, obs=adata_obs.obs.loc[barcode])

            coordinates = adata_train.obs
            coordinates["barcode"] = coordinates.index.values
            coordinates["sampleID"] = sample

            if "neighbors" in self.cell_context:
                neigh = NearestNeighbors(radius=self.radius_neighbors)
                neigh.fit(coordinates[["x_pixel", "y_pixel"]].values)
                neighbors = neigh.radius_neighbors(
                    coordinates[["x_pixel", "y_pixel"]].values, return_distance=True, sort_results=True)[1]
                neighbors = [n[1:32] for n in neighbors]  # remove the cell itself

                cell_ids = coordinates.barcode.values
                # Generate neighbors list as strings of formatted cell IDs
                coordinates["neighbors"] = [
                    "___".join(f"{cell_ids[cell_id]}_{sample}" for cell_id in neigh_ids)
                    for neigh_ids in neighbors
                ]
                max_n_neighbors = coordinates["neighbors"].apply(lambda x: len(x.split("___"))).max()
                if self.max_n_neighbors < max_n_neighbors:
                    self.max_n_neighbors = max_n_neighbors

            coordinates.index = [f"{i}_{sample}" for i in adata_train.obs_names]
            coordinates_df.append(coordinates)

        self.coordinates_df = pd.concat(coordinates_df)

        assert (transcriptomics.index == self.coordinates_df.index).all()

        if self.smooth_n > 0 or self.resolution > 0:

            res = self.resolution if self.resolution > 0 else 1

            resampled_idx, transcriptomics_smooth = spatial_upsample_and_smooth(
                transcriptomics.values,
                self.coordinates_df,
                transcriptomics.index,
                res,
                self.smooth_n,
                self.augmentation)

            if self.smooth_n > 0:
                print(f"Smoothing with n={self.smooth_n}")
                transcriptomics[:] = transcriptomics_smooth

            if self.resolution > 0:
                print(f'Resampling with resolution {self.resolution}...')
                self.old_idx = transcriptomics.index
                transcriptomics = transcriptomics.loc[resampled_idx]
                self.coordinates_df = self.coordinates_df.loc[resampled_idx]
                assert (transcriptomics.index == self.coordinates_df.index).all()

        if self.resample_samples:
            assert (transcriptomics.index == self.coordinates_df.index).all()

            n_count = np.max(self.coordinates_df.sampleID.value_counts()).astype(int)

            org_index = transcriptomics.index.values
            temp_idx = np.array([f"{i}+++{idx}" for i, idx in enumerate(transcriptomics.index)])
            transcriptomics.index = temp_idx
            self.coordinates_df.index = temp_idx

            resampled_idx = get_balanced_index(temp_idx, self.coordinates_df.sampleID, n_count)

            transcriptomics = transcriptomics.loc[resampled_idx]
            self.coordinates_df = self.coordinates_df.loc[resampled_idx]

            new_idx = [i.split("+++")[1] for i in transcriptomics.index]
            transcriptomics.index = new_idx
            self.coordinates_df.index = new_idx

            assert (transcriptomics.index == self.coordinates_df.index).all()

        self.image_feature_source = f"{self.out_folder}/data/image_features/{self.morphology_model_name}"

        if self.scaler is not None:
            transcriptomics.values[:] = self.scaler.transform(transcriptomics.values)
        elif self.normalize is None:
            pass
        elif self.normalize == 'standard':
            self.scaler = StandardScaler()
            transcriptomics.values[:] = self.scaler.fit_transform(transcriptomics.values)
        elif self.normalize == 'robust':
            self.StandardScaler = RobustScaler()
            transcriptomics.values[:] = self.scaler.fit_transform(transcriptomics.values)

        self.transcriptomics_df = transcriptomics
        self.transcriptomics = {b: t.values for b, t in transcriptomics.iterrows()}

    def __len__(self):

        return len(self.coordinates_df)

    def _load_patch(self, sampleID, cell_id):
        X = np.load(f"{self.image_feature_source}_{self.cell_diameter}/{sampleID}/{cell_id}.npy")
        return X

    def __getitem__(self, idx):

        cell_info = self.coordinates_df.iloc[idx]

        X = self._load_patch(cell_info.sampleID, cell_info.barcode)
        X = X[None, :]
        if self.cell_context == 'cell':
            X_cell = X.astype(np.float32)
            X = (X_cell)
        elif self.cell_context == 'cell_neighbors':
            X_cell = X.astype(np.float32)

            X_neighbors = []
            neighbors = cell_info.neighbors.split("___")
            neighbors = [n.split("_")[0] for n in neighbors if n.split("_")[0] != '']

            for cell_neighbor_barcode in neighbors:
                X_neighbor = self._load_patch(cell_info.sampleID, cell_neighbor_barcode)
                X_neighbors.append(X_neighbor)

            X_neighbors = np.array(X_neighbors).astype(np.float32)
            X_neighbors = X_neighbors.reshape(-1, X_cell.shape[1])
            X_neighbors = add_zero_padding(X_neighbors, self.max_n_neighbors)
            X = (X_cell, X_neighbors)

        y = self.transcriptomics[f"{cell_info.barcode}_{cell_info.sampleID}"]
        y = y.astype(np.float32)

        return X, y
