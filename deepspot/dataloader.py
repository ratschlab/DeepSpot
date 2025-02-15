
from .utils.utils_dataloader import spatial_upsample_and_smooth
from .utils.utils_dataloader import compute_neighbors
from .utils.utils_dataloader import add_zero_padding
from .utils.utils_dataloader import load_data

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


class DeepSpotDataLoader(Dataset):
    def __init__(self,
                 out_folder,
                 samples,
                 genes_to_keep,
                 morphology_model_name,
                 target_sum=10000,
                 resolution=0,
                 spot_context='spot',
                 radius_neighbors=0,
                 augmentation="default",
                 normalize=None,
                 scaler=None,
                 smooth_n=0):
        super().__init__()
        self.out_folder = out_folder
        self.samples = samples
        self.spot_context = spot_context
        self.genes_to_keep = genes_to_keep
        self.target_sum = target_sum
        self.morphology_model_name = morphology_model_name
        self.resolution = resolution
        self.radius_neighbors = radius_neighbors
        self.max_n_neighbors = 0
        self.augmentation = augmentation
        self.smooth_n = smooth_n
        self.cache = {}
        self.normalize = normalize
        self.scaler = scaler
        data = load_data(samples, self.out_folder, load_image_features=False, factor=self.target_sum)
        self.transcriptomics_df = pd.DataFrame(data["y"][:, self.genes_to_keep],
                                               index=data["barcode"])

        coordinates_df = []
        for sample in tqdm(samples):

            adata_path = f"{out_folder}/data/h5ad/{sample}.h5ad"
            adata_obs = sc.read_h5ad(adata_path)

            sample_idx = self.transcriptomics_df.index.to_series().apply(lambda x: x.split("_")[1] == sample)
            counts = self.transcriptomics_df[sample_idx]
            barcode = counts.index.to_series().apply(lambda x: x.split("_")[0])
            adata_train = ad.AnnData(counts.values, obs=adata_obs.obs.loc[barcode])

            coordinates = adata_train.obs
            coordinates["barcode"] = coordinates.index.values
            coordinates["sampleID"] = sample

            if "neighbors" in self.spot_context:
                for _, spot in coordinates.iterrows():
                    coordinates["neighbors"] = compute_neighbors(spot, coordinates, self.radius_neighbors)
                max_n_neighbors = coordinates["neighbors"].apply(lambda x: len(x.split("___"))).max()
                if self.max_n_neighbors < max_n_neighbors:
                    self.max_n_neighbors = max_n_neighbors

            coordinates.index = [f"{i}_{sample}" for i in adata_train.obs_names]
            coordinates_df.append(coordinates)

        self.coordinates_df = pd.concat(coordinates_df)

        assert (self.transcriptomics_df.index == self.coordinates_df.index).all()

        if self.smooth_n > 0 or self.resolution > 0:

            res = self.resolution if self.resolution > 0 else 1

            self.resampled_idx, transcriptomics_smooth = spatial_upsample_and_smooth(
                self.transcriptomics_df.values,
                self.coordinates_df,
                self.transcriptomics_df.index,
                res,
                self.smooth_n,
                self.augmentation)

            if self.smooth_n > 0:
                print(f"Smoothing with n={self.smooth_n}")
                self.transcriptomics_df[:] = transcriptomics_smooth

            if self.resolution > 0:
                print(f'Resampling with resolution {self.resolution}...')
                self.old_idx = self.transcriptomics_df.index
                self.transcriptomics_df = self.transcriptomics_df.loc[self.resampled_idx]
                self.coordinates_df = self.coordinates_df.loc[self.resampled_idx]
                assert (self.transcriptomics_df.index == self.coordinates_df.index).all()

        self.image_feature_source = f"{self.out_folder}/data/image_features/{self.morphology_model_name}"

        if self.normalize is None:
            pass
        elif self.scaler is not None:
            self.transcriptomics_df.values[:] = self.scaler.inverse_transform(self.transcriptomics_df.values)
        elif self.normalize == 'standard':
            self.scaler = StandardScaler()
            self.transcriptomics_df.values[:] = self.scaler.fit_transform(self.transcriptomics_df.values)
        elif self.normalize == 'robust':
            self.scaler = RobustScaler()
            self.transcriptomics_df.values[:] = self.scaler.fit_transform(self.transcriptomics_df.values)

    def __len__(self):

        return len(self.coordinates_df)

    def __getitem__(self, idx):

        if idx not in self.cache:

            spot_info = self.coordinates_df.iloc[idx]

            X = np.load(f"{self.image_feature_source}/{spot_info.barcode}_{spot_info.sampleID}.npy")

            if self.spot_context == 'spot':
                indeces_spot = np.array([0])
                X_spot = X[indeces_spot].astype(np.float32)
                X = (X_spot)
            elif self.spot_context == 'spot_subspot':
                indeces_subspot = np.arange(1, len(X))
                indeces_spot = np.array([0])
                X_spot, X_subspot = X[indeces_spot].astype(np.float32), X[indeces_subspot].astype(np.float32)
                X = (X_spot, X_subspot)

            elif self.spot_context == 'spot_neighbors':
                indeces_spot = np.array([0])
                X_spot = X[indeces_spot].astype(np.float32)

                X_neighbors = []
                neighbors = spot_info.neighbors.split("___")
                for spot_neighbor_barcode in neighbors:
                    X_neighbor = np.load(
                        f"{self.image_feature_source}/{spot_neighbor_barcode}_{spot_info.sampleID}.npy")
                    X_neighbors.append(X_neighbor[indeces_spot])

                X_neighbors = np.array(X_neighbors).astype(np.float32)
                X_neighbors = X_neighbors.reshape(-1, X_spot.shape[1])
                X_neighbors = add_zero_padding(X_neighbors, self.max_n_neighbors)
                X = (X_spot, X_neighbors)

            elif self.spot_context == 'spot_subspot_neighbors':
                indeces_subspot = np.arange(1, len(X))
                indeces_spot = np.array([0])
                X_subspot = X[indeces_subspot].astype(np.float32)
                X_spot = X[indeces_spot].astype(np.float32)

                X_neighbors = []
                neighbors = spot_info.neighbors.split("___")
                for spot_neighbor_barcode in neighbors:
                    X_neighbor = np.load(
                        f"{self.image_feature_source}/{spot_neighbor_barcode}_{spot_info.sampleID}.npy")
                    X_neighbors.append(X_neighbor[indeces_spot])

                X_neighbors = np.array(X_neighbors).astype(np.float32)
                X_neighbors = X_neighbors.reshape(-1, X_subspot.shape[1])
                X_neighbors = add_zero_padding(X_neighbors, self.max_n_neighbors)
                X = (X_spot, X_subspot, X_neighbors)

            y = self.transcriptomics_df.iloc[idx].values
            y = y.astype(np.float32)

            self.cache[idx] = [X, y]

        else:
            X, y = self.cache[idx]

        return X, y
