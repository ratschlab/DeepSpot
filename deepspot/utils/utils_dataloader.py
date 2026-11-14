from matplotlib import pyplot as plt
from torch.backends import cudnn
import numpy as np
import random
import torch
import os

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm
import anndata as ad
import lightning as L
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


BALANCE_MODES = ("oversample", "bootstrap")


def get_balanced_index(barcode, labels, n_count, mode="bootstrap"):
    """Cluster-balanced list of one sample's spots.

    ``"oversample"``: every spot once, plus random draws (with replacement) of
    each cluster's own spots until it has ``n_count`` (the largest cluster's
    size), so no spot is left out. ``"bootstrap"`` (the original): ``n_count - 1``
    draws with replacement per cluster, which leaves out part of every cluster
    (about 1/3 of the largest one).
    """
    assert mode in BALANCE_MODES, f"mode must be one of {BALANCE_MODES}"
    labels = np.array(labels)
    resampled_barcodes = []

    for label in np.unique(labels):
        members = barcode[labels == label]
        if mode == "oversample":
            resampled_barcodes.extend(members)
            resampled_barcodes.extend(np.random.choice(members, size=max(n_count - len(members), 0)))
        else:
            resampled_barcodes.extend(np.random.choice(members, size=n_count - 1))
    return resampled_barcodes


def run_default(adata, resolution=1.0):
    adata = adata.copy()
    n_comps = min(adata.shape[1] - 1, 50)
    sc.pp.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
    adata.obsm["latent"] = adata.obsm["X_pca"]
    adata.obs["label"] = adata.obs["leiden"].values
    return adata


def run_aestetik(adata, window_size=3, resolution=1.0):
    from aestetik import AESTETIK

    adata = adata.copy()
    sc.pp.pca(adata)

    # we set the transcriptomics modality
    adata.obsm["X_pca_transcriptomics"] = adata.obsm["X_pca"]

    # we set the morphology modality
    adata.obsm["X_pca_morphology"] = np.ones((len(adata), 5))  # dummy number to keep dim low

    resolution = float(resolution)  # leiden with resolution
    model = AESTETIK(clustering_method="leiden",
                     nCluster=resolution,
                     window_size=window_size,
                     morphology_weight=0)

    model.fit(X=adata)
    model.predict(X=adata, cluster=True)
    adata.obsm["latent"] = adata.obsm["AESTETIK"]
    adata.obs["label"] = adata.obs["AESTETIK_cluster"].values
    return adata


def spatial_upsample_and_smooth(counts, obs, barcode, resolution, smooth_n=0, augmentation="default",
                                balance_mode="bootstrap"):
    """Cluster-balance (``resolution`` > 0) and/or smooth (``smooth_n`` > 0) each sample's spots.

    Clusters are ``obs.label`` of :func:`run_default` (Leiden) or :func:`run_aestetik`
    (AESTETIK). ``balance_mode`` is passed to :func:`get_balanced_index`;
    ``"bootstrap"`` reproduces the original behaviour, which sized the clusters by
    ``label`` but drew spots grouped by ``obs.leiden`` (with ``aestetik`` that is a
    different, transcriptomics-only clustering).
    """
    samples = np.array([b.split('_')[1] for b in barcode])
    unqiue_samples = np.unique(samples)
    resampled_barcodes = []
    transcriptomics_smooth = np.zeros(counts.shape)
    for sample in tqdm(unqiue_samples):
        idx = samples == sample
        sample_barcode = barcode[idx]
        adata = ad.AnnData(counts[idx, :], obs=obs.iloc[idx])

        if resolution:
            if augmentation == "aestetik":
                adata = run_aestetik(adata, resolution=resolution)
            elif augmentation == "default":
                adata = run_default(adata, resolution=resolution)
            else:
                # If none of the above conditions are met, raise a NotImplementedError
                raise NotImplementedError(f"Not implemented: {augmentation}")

            # sc.pl.umap(adata, color='leiden')
            # most_common, num_most_common = Counter(adata.obs.label).most_common(1)[0]
            n_count = np.max(adata.obs.label.value_counts()).astype(int)
            clusters = adata.obs.label if balance_mode == "oversample" else adata.obs.leiden
            resampled_barcodes.extend(get_balanced_index(sample_barcode, clusters, n_count, mode=balance_mode))
        else:
            resampled_barcodes.extend(sample_barcode)

        if smooth_n > 0:
            neigh = NearestNeighbors(n_neighbors=smooth_n)
            neigh.fit(adata.obsm["latent"])
            neigh_idx = neigh.kneighbors(adata.obsm["latent"], return_distance=False)
            smooth_counts = adata.X[neigh_idx].mean(axis=1)
            transcriptomics_smooth[idx, :] = smooth_counts

    return np.array(resampled_barcodes), transcriptomics_smooth


def add_zero_padding(original_array, desired_padding):
    if desired_padding == original_array.shape[0]:
        return original_array
    # Calculate the amount of padding needed
    padding_needed = desired_padding - original_array.shape[0]

    padded_array = np.pad(original_array, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)

    return padded_array


def compute_neighbors(row, coordinates, radius=1):
    query = f"""
    ((x_array - {radius}) <= {row.x_array} <= (x_array + {radius})) and \
    ((y_array - {radius}) <= {row.y_array} <= (y_array + {radius})) and \
    (sampleID == '{row.sampleID}' and barcode != '{row.barcode}')
    """
    neighbors = coordinates.query(query)
    neighbors = neighbors.barcode.values
    neighbors = "___".join(neighbors)
    return neighbors


def compute_all_neighbors(coordinates, radius=1):
    """``compute_neighbors`` for every spot of one sample at once.

    ``coordinates`` holds one sample's spots (``x_array``, ``y_array``,
    ``barcode``); returns, aligned with its rows, the "___"-joined barcodes of
    the other spots within ``radius`` array steps, in row order (the same
    strings as ``compute_neighbors``).
    """
    x = coordinates.x_array.to_numpy()
    y = coordinates.y_array.to_numpy()
    barcode = coordinates.barcode.to_numpy().astype(str)
    out = []
    for start in range(0, len(x), 2048):  # chunks keep the distance matrix small
        xs, ys = x[start:start + 2048, None], y[start:start + 2048, None]
        near = (np.abs(x[None, :] - xs) <= radius) & (np.abs(y[None, :] - ys) <= radius)
        near[np.arange(len(xs)), np.arange(start, start + len(xs))] = False
        out.extend("___".join(barcode[row]) for row in near)
    return out


NEIGHBOR_METHODS = ("spatial", "array")


def spot_pitch(xy):
    """Centre-to-centre spot spacing of one sample (pixels): the 25th percentile of
    the spots' nearest-neighbour distances, ignoring duplicate positions, so a slide
    thinned by QC still gives the lattice spacing. None if it cannot be estimated."""
    from scipy.spatial import cKDTree
    if len(xy) < 2:
        return None
    d, _ = cKDTree(xy).query(xy, k=2)
    d = d[:, 1][d[:, 1] > 0]
    return float(np.percentile(d, 25)) if len(d) else None


def compute_spatial_neighbors(coordinates, radius=1):
    """Neighbours of every spot of one sample, from its pixel coordinates.

    Spots within ``(radius + 0.5) * pitch`` of a spot (see :func:`spot_pitch`) are
    its neighbours: for radius 1, the 6 adjacent spots of a Visium hexagonal grid or
    the 8 surrounding spots of a square (ST) grid, independent of how the array
    coordinates are numbered. Spots at the tissue boundary get fewer neighbours and
    isolated spots none (the model pools only real neighbours; see
    ``DeepSpot.mask_padded_neighbors``). ``coordinates`` holds one sample's spots
    (``x_pixel``, ``y_pixel``, ``barcode``); returns, aligned with its rows, the
    "___"-joined neighbour barcodes, nearest first (the format of
    :func:`compute_neighbors`).
    """
    from scipy.spatial import cKDTree
    xy = coordinates[["x_pixel", "y_pixel"]].to_numpy(dtype=np.float64)
    barcode = coordinates.barcode.to_numpy().astype(str)
    pitch = spot_pitch(xy)
    if pitch is None or radius <= 0:
        return [""] * len(xy)
    tree = cKDTree(xy)
    out = []
    for i, near in enumerate(tree.query_ball_point(xy, (radius + 0.5) * pitch)):
        near = np.array([j for j in near if j != i], dtype=np.int64)
        dist = np.hypot(*(xy[near] - xy[i]).T) if len(near) else np.empty(0)
        out.append("___".join(barcode[near[np.lexsort((near, dist))]]))
    return out


def load_multiple_pickles(files):
    return pd.concat([pd.read_pickle(f) for f in files])


def log1p_normalization(arr, factor=10000, eps=1e-12):
    return np.log1p(((arr.T / (np.sum(arr, axis=1) + eps)).T) * factor)


def read_h5ad_counts(path):
    """Counts (CSR float32), spot ids and gene names of an h5ad, read without its obs columns or uns."""
    import h5py
    import scipy.sparse as sp
    with h5py.File(path, "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Group):
            matrix = {"csr_matrix": sp.csr_matrix, "csc_matrix": sp.csc_matrix}[X.attrs["encoding-type"]]
            X = matrix((X["data"][:], X["indices"][:], X["indptr"][:]), shape=tuple(X.attrs["shape"]))
        else:
            X = X[:]
        obs_names = f["obs"][f["obs"].attrs["_index"]].asstr()[:]
        var_names = f["var"][f["var"].attrs["_index"]].asstr()[:]
    return sp.csr_matrix(X, dtype=np.float32), obs_names, var_names


def load_counts(samples, out_folder):
    """Raw counts of ``samples``: DataFrame (spots x genes of info_highly_variable_genes.csv).

    Reads dense ``data/inputX/{sample}.pkl`` files when every sample has one;
    otherwise ``data/h5ad/{sample}.h5ad`` (he2st visium-v3), with each
    sample's genes placed in the columns of
    ``{out_folder}/info_highly_variable_genes.csv`` (unmeasured genes are 0).
    """
    pickles = [f"{out_folder}/data/inputX/{sample}.pkl" for sample in samples]
    h5ads = [f"{out_folder}/data/h5ad/{sample}.h5ad" for sample in samples]
    # a pickle older than its h5ad is a leftover of a previous build
    stale = [p for p, h in zip(pickles, h5ads)
             if os.path.exists(p) and os.path.exists(h) and os.path.getmtime(p) < os.path.getmtime(h)]
    assert not stale, f"{len(stale)} inputX pickles older than their h5ad (e.g. {stale[0]}); remove data/inputX"
    if all(os.path.exists(f) for f in pickles):
        return load_multiple_pickles(pickles)
    genes = pd.Index(pd.read_csv(f"{out_folder}/info_highly_variable_genes.csv", usecols=["gene_name"]).gene_name)
    parts = [read_h5ad_counts(h) for h in h5ads]
    dense = np.zeros((sum(X.shape[0] for X, _, _ in parts), len(genes)), dtype=np.float32)
    row = 0
    for X, _, var_names in parts:
        col = genes.get_indexer(var_names)
        assert (col >= 0).all(), "h5ad genes missing from info_highly_variable_genes.csv"
        X = X.tocoo()
        dense[row + X.row, col[X.col]] = X.data
        row += X.shape[0]
    index = np.concatenate([obs_names for _, obs_names, _ in parts])
    return pd.DataFrame(dense, index=index, columns=genes.values, copy=False)


def load_data(
        samples,
        out_folder,
        feature_model=None,
        cell_diameter=None,
        load_image_features=True,
        factor=10000,
        raw_counts=False):
    gene_expr = load_counts(samples, out_folder)

    barcode_list = gene_expr.index.values
    gene_expr = gene_expr.values

    if load_image_features and cell_diameter:
        image_features_files = [
            f"{out_folder}/data/image_features/{feature_model}_{cell_diameter}/{sample}.pkl" for sample in samples]
        image_features_emb = load_multiple_pickles(image_features_files)
        image_features_emb = image_features_emb.loc[barcode_list]

    elif load_image_features:

        image_features_files = [f"{out_folder}/data/image_features/{feature_model}/{sample}.pkl" for sample in samples]
        image_features_emb = load_multiple_pickles(image_features_files)
        image_features_emb = image_features_emb.loc[barcode_list]

    data = {}

    if not raw_counts:
        gene_expr = log1p_normalization(gene_expr, factor=factor)

    data["y"] = gene_expr

    if load_image_features:
        data["X"] = image_features_emb

    data["barcode"] = barcode_list
    return data
