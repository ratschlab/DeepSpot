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


def get_balanced_index(barcode, labels, n_count):
    labels = np.array(labels)
    resampled_barcodes = []

    for label in np.unique(labels):
        resampled_barcodes.extend(np.random.choice(barcode[labels == label], size=n_count - 1))
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
    model = AESTETIK(adata,
                     clustering_method="leiden",
                     nCluster=resolution,
                     window_size=window_size,
                     morphology_weight=0)

    model.prepare_input_for_model()
    model.train()
    model.compute_spot_representations(cluster=True, n_repeats=10)
    adata = model.adata.copy()
    adata.obsm["latent"] = adata.obsm["AESTETIK"]
    adata.obs["label"] = adata.obs["AESTETIK_cluster"].values
    return adata


def spatial_upsample_and_smooth(counts, obs, barcode, resolution, smooth_n=0, augmentation="default"):
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
            resampled_barcodes.extend(get_balanced_index(sample_barcode, adata.obs.leiden, n_count))  # num_most_common
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


def load_multiple_pickles(files):
    return pd.concat([pd.read_pickle(f) for f in files])


def log1p_normalization(arr, factor=10000):
    # max_vals = arr.max(axis=1, keepdims=True)
    return np.log1p((arr.T / np.sum(arr, axis=1)).T * factor)


def load_data(
        samples,
        out_folder,
        feature_model=None,
        cell_diameter=None,
        load_image_features=True,
        factor=10000,
        raw_counts=False):
    barcode_list = []
    image_features_emb = []
    gene_expr = []

    expr_files = [f"{out_folder}/data/inputX/{sample}.pkl" for sample in samples]

    gene_expr = load_multiple_pickles(expr_files)

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
