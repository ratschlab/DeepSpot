from deepspot.utils.utils_dataloader import spatial_upsample_and_smooth
from deepspot.utils.utils_dataloader import compute_all_neighbors
from deepspot.utils.utils_dataloader import compute_spatial_neighbors
from deepspot.utils.utils_dataloader import NEIGHBOR_METHODS
from deepspot.utils.utils_dataloader import BALANCE_MODES
from deepspot.utils.utils_dataloader import add_zero_padding
from deepspot.utils.utils_dataloader import load_data

from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

from tqdm import tqdm
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np
import torch


class DeepSpotDataLoader(Dataset):
    """Spot image features (spot, sub-spots, neighbours) and expression for DeepSpot.

    With ``preload_features=True`` (default) every spot's feature file
    (``data/image_features/{model}/{sample}/{barcode}.npy``) is read once, at
    construction, into one float32 array that DataLoader workers share;
    otherwise each item reads its files (and caches the item) on access.
    Both give the same items. With preloaded features, :meth:`collate_batch`
    builds a whole batch by array indexing (same tensors as the default collate
    of the items); use it as ``collate_fn`` of a DataLoader over
    ``range(len(dataset))``.

    ``neighbor_method``: ``"spatial"`` (default) takes as neighbours the spots
    within ``radius_neighbors + 0.5`` spot spacings in pixel space
    (:func:`~deepspot.utils.utils_dataloader.compute_spatial_neighbors`: the 6
    adjacent Visium spots for radius 1); ``"array"`` is the original square
    window in array coordinates, which on Visium's hexagonal grid finds only
    the 4 diagonal neighbours (kept to reproduce older models).

    ``balance_mode`` (with ``resolution`` > 0): ``"oversample"`` (default) keeps
    every training spot and oversamples the smaller clusters to the size of the
    largest; ``"bootstrap"`` is the original fixed bootstrap sample, which leaves
    out a part of every cluster (see utils_dataloader.get_balanced_index).
    """

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
                 smooth_n=0,
                 preload_features=True,
                 neighbor_method="spatial",
                 balance_mode="oversample"):
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
        assert neighbor_method in NEIGHBOR_METHODS, f"neighbor_method must be one of {NEIGHBOR_METHODS}"
        self.neighbor_method = neighbor_method
        assert balance_mode in BALANCE_MODES, f"balance_mode must be one of {BALANCE_MODES}"
        self.balance_mode = balance_mode
        data = load_data(samples, self.out_folder, load_image_features=False, factor=self.target_sum)
        self.transcriptomics_df = pd.DataFrame(data["y"][:, self.genes_to_keep],
                                               index=data["barcode"])
        del data

        # spot ids are "{barcode}_{sample}"
        spot_sample = self.transcriptomics_df.index.str.split("_").str[1]
        coordinates_df = []
        for sample in tqdm(samples):

            adata_path = f"{out_folder}/data/h5ad/{sample}.h5ad"
            obs = sc.read_h5ad(adata_path, backed="r").obs
            # he2st visium-v3 h5ads are indexed by "{barcode}_{sample}" and keep the
            # raw barcode in obs["barcode"]; older h5ads are indexed by raw barcode.
            if "barcode" in obs.columns:
                obs = obs.set_index("barcode", drop=False)
                obs.index.name = None

            counts = self.transcriptomics_df[spot_sample == sample]
            barcode = counts.index.str.split("_").str[0]
            adata_train = ad.AnnData(counts.values, obs=obs.loc[barcode])

            coordinates = adata_train.obs
            coordinates["barcode"] = coordinates.index.values
            coordinates["sampleID"] = sample

            if "neighbors" in self.spot_context:
                find = compute_spatial_neighbors if self.neighbor_method == "spatial" else compute_all_neighbors
                coordinates["neighbors"] = find(coordinates, self.radius_neighbors)
                max_n_neighbors = coordinates["neighbors"].apply(lambda x: len(x.split("___"))).max()
                if self.max_n_neighbors < max_n_neighbors:
                    self.max_n_neighbors = max_n_neighbors

            coordinates.index = [f"{i}_{sample}" for i in adata_train.obs_names]
            coordinates_df.append(coordinates)

        self.coordinates_df = pd.concat(coordinates_df)

        assert (self.transcriptomics_df.index == self.coordinates_df.index).all()
        # every spot of the samples (resampling below may drop spots that are still neighbours)
        self._spots = self.coordinates_df[["sampleID", "barcode"]].copy()

        if self.smooth_n > 0 or self.resolution > 0:

            res = self.resolution if self.resolution > 0 else 1

            self.resampled_idx, transcriptomics_smooth = spatial_upsample_and_smooth(
                self.transcriptomics_df.values,
                self.coordinates_df,
                self.transcriptomics_df.index,
                res,
                self.smooth_n,
                self.augmentation,
                balance_mode=self.balance_mode)

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

        self.features = None
        if preload_features:
            self._preload_features()

    def _preload_features(self, n_threads=8):
        """Read each spot's features once; map items and their neighbours to rows."""
        keys = list(zip(self._spots.sampleID, self._spots.barcode))
        row = {k: i for i, k in enumerate(keys)}
        first = self._load_patch(*keys[0])
        self.features = np.empty((len(keys), *first.shape), dtype=np.float32)

        def fill(i):
            self.features[i] = self._load_patch(*keys[i])

        with ThreadPoolExecutor(n_threads) as pool:
            list(tqdm(pool.map(fill, range(len(keys))), total=len(keys), desc="image features"))

        sample_ids = self.coordinates_df.sampleID.values
        self.feature_row = np.array([row[k] for k in zip(sample_ids, self.coordinates_df.barcode.values)])
        if "neighbors" in self.spot_context:
            self.neighbor_rows = [np.array([row[(s, n)] for n in neighbors.split("___") if n], dtype=np.int64)
                                  for s, neighbors in zip(sample_ids, self.coordinates_df.neighbors.values)]
            # items x max_n_neighbors feature rows, -1 = zero padding (as add_zero_padding)
            self.neighbor_matrix = np.full((len(self.neighbor_rows), max(self.max_n_neighbors, 1)), -1, dtype=np.int64)
            for i, r in enumerate(self.neighbor_rows):
                self.neighbor_matrix[i, :len(r)] = r

    def collate_batch(self, indices):
        """``default_collate([self[i] for i in indices])``, built by array indexing (preloaded features only)."""
        assert self.features is not None, "collate_batch needs preload_features=True"
        idx = np.asarray(indices, dtype=np.int64)
        X = self.features[self.feature_row[idx]]
        x_spot = torch.from_numpy(np.ascontiguousarray(X[:, :1]))  # contiguous, as default_collate's stack
        y = torch.from_numpy(self.transcriptomics_df.values[idx].astype(np.float32))
        if self.spot_context == 'spot':
            return [x_spot, y]
        parts = [x_spot]
        if "subspot" in self.spot_context:
            parts.append(torch.from_numpy(np.ascontiguousarray(X[:, 1:])))
        if "neighbors" in self.spot_context:
            m = self.neighbor_matrix[idx]
            nb = self.features[np.maximum(m, 0), 0]
            nb[m < 0] = 0
            parts.append(torch.from_numpy(nb))
        return [parts, y]

    def __len__(self):

        return len(self.coordinates_df)

    def _load_patch(self, sampleID, cell_id):
        X = np.load(f"{self.image_feature_source}/{sampleID}/{cell_id}.npy")
        return X

    def _item_features(self, idx):
        """(spot features, list of neighbour spot features) of item ``idx``."""
        if self.features is not None:
            neighbors = self.neighbor_rows[idx] if "neighbors" in self.spot_context else []
            return self.features[self.feature_row[idx]], [self.features[i] for i in neighbors]
        spot_info = self.coordinates_df.iloc[idx]
        X = self._load_patch(spot_info.sampleID, spot_info.barcode)
        neighbors = []
        if "neighbors" in self.spot_context:
            neighbors = [self._load_patch(spot_info.sampleID, n) for n in spot_info.neighbors.split("___") if n]  # remove 'empty' neighbors
        return X, neighbors

    def __getitem__(self, idx):

        if idx in self.cache:
            X, y = self.cache[idx]
            return X, y

        X, neighbors = self._item_features(idx)
        indeces_spot = np.array([0])

        if self.spot_context == 'spot':
            X = X[indeces_spot].astype(np.float32)
        elif self.spot_context == 'spot_subspot':
            indeces_subspot = np.arange(1, len(X))
            X = (X[indeces_spot].astype(np.float32), X[indeces_subspot].astype(np.float32))

        elif self.spot_context == 'spot_neighbors':
            X_spot = X[indeces_spot].astype(np.float32)
            X_neighbors = np.array([n[indeces_spot] for n in neighbors]).astype(np.float32)
            X_neighbors = X_neighbors.reshape(-1, X_spot.shape[1])
            X_neighbors = add_zero_padding(X_neighbors, self.max_n_neighbors)
            X = (X_spot, X_neighbors)

        elif self.spot_context == 'spot_subspot_neighbors':
            indeces_subspot = np.arange(1, len(X))
            X_subspot = X[indeces_subspot].astype(np.float32)
            X_spot = X[indeces_spot].astype(np.float32)
            X_neighbors = np.array([n[indeces_spot] for n in neighbors]).astype(np.float32)
            X_neighbors = X_neighbors.reshape(-1, X_subspot.shape[1])
            X_neighbors = add_zero_padding(X_neighbors, self.max_n_neighbors)
            X = (X_spot, X_subspot, X_neighbors)

        y = self.transcriptomics_df.iloc[idx].values
        y = y.astype(np.float32)

        if self.features is None:  # file-backed: keep the old per-item cache
            self.cache[idx] = [X, y]
        return X, y
