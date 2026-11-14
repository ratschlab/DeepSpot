from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from huggingface_hub import login, hf_hub_download
from transformers import AutoImageProcessor, ViTModel
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import numbers
import json
import timm
import glob
import math
import sys
import os
import threading

from .utils_dataloader import compute_neighbors
from .utils_dataloader import compute_spatial_neighbors

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


def get_morphology_model_and_preprocess(model_name, device, model_path=None):

    if model_name == "uni":

        morphology_model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True)
        morphology_model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        morphology_model.eval()

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224, antialias=True),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        feature_dim = 1024
    elif model_name == "hoptimus0":

        morphology_model = timm.create_model(
            "vit_giant_patch14_reg4_dinov2",
            img_size=224,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True
        )
        morphology_model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        morphology_model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])

        feature_dim = 1536

    elif model_name == "phikon":

        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model = ViTModel.from_pretrained(model_path, add_pooling_layer=False)

        def image_processor_edit(x):
            x = image_processor(x, return_tensors="pt")["pixel_values"].squeeze()
            return x

        class MyModel(torch.nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                with torch.no_grad():
                    x = self.model(x)
                    x = x.last_hidden_state[:, 0, :]
                    return x

        preprocess = image_processor_edit
        morphology_model = MyModel(model)
        feature_dim = 768

    elif model_name == "phikonv2":

        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)

        def image_processor_edit(x):
            x = image_processor(x, return_tensors="pt")["pixel_values"].squeeze()
            return x

        class MyModel(torch.nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                with torch.no_grad():
                    x = self.model(x)
                    x = x.last_hidden_state[:, 0, :]
                    return x

        preprocess = image_processor_edit
        morphology_model = MyModel(model)
        feature_dim = 1024

    elif model_name == "inception":

        weights = Inception_V3_Weights.DEFAULT
        morphology_model = inception_v3(weights=weights)
        morphology_model.fc = torch.nn.Identity()

        morphology_model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            weights.transforms(antialias=True),
        ])

        feature_dim = 2048

    elif model_name == "resnet50":

        weights = ResNet50_Weights.DEFAULT
        morphology_model = resnet50(weights=weights)
        morphology_model.fc = torch.nn.Identity()

        morphology_model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            weights.transforms(antialias=True),
        ])

        feature_dim = 2048

    elif model_name == "densenet121":

        weights = DenseNet121_Weights.DEFAULT
        morphology_model = densenet121(weights=weights)
        morphology_model.classifier = torch.nn.Identity()

        morphology_model.eval()

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            weights.transforms(antialias=True),
        ])

        feature_dim = 1024

    return morphology_model, preprocess, feature_dim


def get_low_res_image(image_path, downsample_factor):
    import pyvips
    image = pyvips.Image.new_from_file(image_path, access='sequential')
    image_low_res = image.resize(1 / downsample_factor)
    image_low_res_arr = np.ndarray(buffer=image_low_res.write_to_memory(),
                                   dtype=format_to_dtype[image_low_res.format],
                                   shape=[image_low_res.height, image_low_res.width, image_low_res.bands])
    return image_low_res_arr


def crop_tile(image, x_pixel, y_pixel, cell_diameter):
    x = x_pixel - int(cell_diameter // 2)
    y = y_pixel - int(cell_diameter // 2)
    cell = image.crop(y, x, cell_diameter, cell_diameter)
    main_tile = np.ndarray(buffer=cell.write_to_memory(),
                           dtype=format_to_dtype[cell.format],
                           shape=[cell.height, cell.width, cell.bands])
    main_tile = main_tile[:, :, :3]
    return main_tile


def compute_mini_tiles(image, n_tiles, super_resolution_mode=False):
    D = image.shape[0]  # assuming the image is a square, so width = height = D
    n = int(np.sqrt(n_tiles))  # number of squares along one dimension
    square_size = D // n
    D, n, square_size

    # List to hold the split images
    squares = []

    if not super_resolution_mode:
        # Loop to crop the image into n x n squares
        for i in range(n):
            for j in range(n):
                left = j * square_size
                right = left + square_size

                lower = i * square_size
                upper = lower + square_size

                # Crop the image
                crop = image[lower:upper, left:right, ]
                squares.append(crop)
    else:

        left = (square_size // n)
        right = left + square_size

        lower = (square_size // n)
        upper = lower + square_size

        crop = image[lower:upper, left:right, ]
        squares.append(crop)

    return squares


def detach_and_convert(data):
    return data[None, ].detach().float()


# Tiles per morphology-model forward pass: on the GPU 32 spots x (1 + 9 mini
# tiles), as in the per-spot feature extractor; on the CPU large batches run
# slower than small ones.
PREDICT_BATCH_TILES = 320
PREDICT_BATCH_TILES_CPU = 16


def predict_batch_tiles(device):
    return PREDICT_BATCH_TILES if torch.device(device).type == "cuda" else PREDICT_BATCH_TILES_CPU


class SpotTileDataset(torch.utils.data.Dataset):
    """Preprocessed tiles of one spot per item, cropped in :func:`tile_loader`'s threads.

    Item ``i`` stacks ``preprocess`` of the tile of spot ``i`` (``tiles="spot"``)
    or of its mini tiles as ``compute_mini_tiles`` cuts them
    (``tiles="subspot"``), as float32 (n_tiles, C, H, W). With
    ``zero_tile_of=j`` one extra last item holds the zero image that stands in
    for the neighbours of a spot that has none, shaped like the tile of spot
    ``j``. The slide is opened once, on first use; its threads share it.
    """

    def __init__(self, image_path, x_pixel, y_pixel, diameter, preprocess, tiles="spot",
                 n_mini_tiles=None, super_resolution=False, zero_tile_of=None):
        self.image_path = image_path
        self.x_pixel = x_pixel
        self.y_pixel = y_pixel
        self.diameter = diameter
        self.preprocess = preprocess
        self.tiles = tiles
        self.n_mini_tiles = n_mini_tiles
        self.super_resolution = super_resolution
        self.zero_tile_of = zero_tile_of
        self.image = None
        self._open_lock = threading.Lock()

    def __len__(self):
        return len(self.x_pixel) + (self.zero_tile_of is not None)

    def __getitem__(self, i):
        if self.image is None:
            with self._open_lock:
                if self.image is None:
                    import pyvips
                    self.image = pyvips.Image.new_from_file(self.image_path)
        j = self.zero_tile_of if i == len(self.x_pixel) else i
        tile = crop_tile(self.image, self.x_pixel[j], self.y_pixel[j], self.diameter)
        if j != i:
            tiles = np.zeros((1, *tile.shape))
        elif self.tiles == "spot":
            tiles = [tile]
        else:
            tiles = compute_mini_tiles(tile, self.n_mini_tiles, self.super_resolution)
        return torch.stack([self.preprocess(t) for t in tiles]).float()


class _ThreadedBatches:
    """Ordered batches (``torch.stack`` of the items) of ``dataset``, read one batch ahead."""

    def __init__(self, dataset, batch_size, n_threads, pin_memory):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.pin_memory = pin_memory

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        with ThreadPoolExecutor(self.n_threads) as pool:
            pending, submitted = deque(), 0
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                while submitted < min(stop + self.batch_size, n):
                    pending.append(pool.submit(self.dataset.__getitem__, submitted))
                    submitted += 1
                X = torch.stack([pending.popleft().result() for _ in range(start, stop)])
                yield X.pin_memory() if self.pin_memory else X


def tile_loader(dataset, batch_size, device):
    """Ordered batches of ``dataset`` like a DataLoader's, read by one thread per CPU of the job.

    Threads, not DataLoader worker processes: a forked worker deadlocks once the
    caller has used libvips (e.g. a notebook that opened the slide with pyvips),
    and spawned workers would re-run a calling script that has no ``__main__``
    guard. libvips and torch release the GIL while they crop and preprocess.
    """
    n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return _ThreadedBatches(dataset, batch_size, n_cpus, pin_memory=torch.device(device).type == "cuda")


def spot_neighbor_positions(obs, radius=1):
    """Positions in ``obs`` of the neighbours of every spot, in ``obs`` order.

    The spots that ``compute_neighbors(spot, obs, radius)`` followed by
    ``obs.query('barcode in @neighbors_barcodes')`` select: same ``sampleID``,
    within ``radius`` steps in both ``x_array`` and ``y_array``, the spot
    itself excluded. Integer array coordinates and unique plain-string
    barcodes (Visium and TCGA spots) are matched on a grid lookup; anything
    else runs that per-spot query.
    """
    sample_ids = obs["sampleID"].tolist()
    barcodes = obs["barcode"].tolist()
    plain = [s for s in sample_ids + barcodes if isinstance(s, str) and "'" not in s and "\\" not in s]
    integer_grid = all(isinstance(obs[c].dtype, np.dtype) and obs[c].dtype.kind in "iu"
                       for c in ("x_array", "y_array"))
    grid = (integer_grid and isinstance(radius, numbers.Real) and math.isfinite(radius) and radius < 64
            and len(plain) == 2 * len(obs) and len(set(barcodes)) == len(obs)
            and not any("___" in b for b in barcodes))
    if not grid:
        positions = []
        for _, spot in obs.iterrows():
            neighbors_barcodes = compute_neighbors(spot, obs, radius=radius).split('___')
            positions.append(np.flatnonzero(obs["barcode"].isin(neighbors_barcodes).to_numpy()))
        return positions

    r = math.floor(radius)
    offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
    keys = list(zip(sample_ids, obs["x_array"].tolist(), obs["y_array"].tolist()))
    spots_at = {}
    for i, key in enumerate(keys):
        spots_at.setdefault(key, []).append(i)
    positions = []
    for i, (sample_id, x, y) in enumerate(keys):
        found = [j for dx, dy in offsets for j in spots_at.get((sample_id, x + dx, y + dy), ()) if j != i]
        positions.append(np.array(sorted(found), dtype=np.int64))
    return positions


def model_neighbor_positions(obs, model_expression, radius=1):
    """Neighbour positions of every spot, defined as for the model's training data.

    Models trained with ``neighbor_method="spatial"`` (DeepSpot.neighbor_method)
    use :func:`compute_spatial_neighbors` per sample; older models (no attribute)
    the array-coordinate window of :func:`spot_neighbor_positions`.
    """
    if getattr(model_expression, "neighbor_method", "array") != "spatial":
        return spot_neighbor_positions(obs, radius=radius)
    position = {(s, b): i for i, (s, b) in enumerate(zip(obs["sampleID"], obs["barcode"].astype(str)))}
    positions = [None] * len(obs)
    for sample_id, rows in obs.groupby("sampleID", sort=False).indices.items():
        sub = obs.iloc[rows]
        for i, near in zip(rows, compute_spatial_neighbors(sub, radius=radius)):
            positions[i] = np.array([position[(sample_id, b)] for b in near.split("___") if b], dtype=np.int64)
    return positions


def predict_spot_spatial_transcriptomics_from_image_path(image_path,
                                                         adata,
                                                         spot_diameter,
                                                         n_mini_tiles,
                                                         preprocess,
                                                         morphology_model,
                                                         model_expression,
                                                         device,
                                                         super_resolution=False,
                                                         return_std=False,
                                                         neighbor_radius=1):
    """Predict the expression of every spot of ``adata`` from the slide image.

    ``model_expression`` gets, per spot, the embeddings of the spot tile, of its
    mini tiles and of the tiles of its neighbours (within ``neighbor_radius``,
    defined as in the model's training data, see :func:`model_neighbor_positions`;
    a spot without neighbours gets an all-zero row if the model masks padding,
    else the embedding of a zero image). Each tile is embedded
    once, in batches of ``predict_batch_tiles(device)``, with crops and preprocessing in
    :func:`tile_loader`'s threads: first all spot tiles (a neighbour's embedding is the
    embedding of that spot), then the mini tiles, each batch of which goes
    through ``model_expression`` grouped by neighbour count.

    Returns the inverse-transformed ``counts`` (spots x genes) and, if
    ``return_std``, the ensemble std as the model outputs it. Equal to the
    former per-spot loop up to floating-point rounding.
    """
    obs = adata.obs
    n_spots = len(obs)
    if n_spots == 0:
        counts = np.array([]).squeeze()
        return (counts, np.array([]).squeeze()) if return_std else counts

    x_pixel, y_pixel = obs["x_pixel"].tolist(), obs["y_pixel"].tolist()
    neighbors = model_neighbor_positions(obs, model_expression, radius=neighbor_radius)
    # A model that masks neighbour padding gets an all-zero neighbour row for a spot
    # without neighbours (it then uses the spot's own embedding, as in training);
    # older models get the embedding of a zero image, as before.
    masks_padding = getattr(model_expression, "mask_padded_neighbors", False)
    isolated = [] if masks_padding else [i for i, positions in enumerate(neighbors) if len(positions) == 0]
    n_subspot_tiles = 1 if super_resolution else int(np.sqrt(n_mini_tiles)) ** 2
    batch_tiles = predict_batch_tiles(device)

    counts = std_counts = None
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            # Spot tiles, plus the zero image last if any spot has no neighbours.
            spot_tiles = SpotTileDataset(image_path, x_pixel, y_pixel, spot_diameter, preprocess, tiles="spot",
                                         zero_tile_of=isolated[0] if isolated else None)
            spot_features = []
            for X in tqdm(tile_loader(spot_tiles, batch_tiles, device), desc="spot tiles"):
                X = morphology_model(X.to(device, non_blocking=True).flatten(0, 1))
                spot_features.append(X.detach().float().cpu())
            spot_features = torch.cat(spot_features)
            zero_features = spot_features[n_spots:]
            spot_features = spot_features[:n_spots]

            # Mini tiles, then the expression model on the assembled inputs.
            subspot_tiles = SpotTileDataset(image_path, x_pixel, y_pixel, spot_diameter, preprocess,
                                            tiles="subspot", n_mini_tiles=n_mini_tiles,
                                            super_resolution=super_resolution)
            batch_spots = max(1, batch_tiles // n_subspot_tiles)
            start = 0
            for X in tqdm(tile_loader(subspot_tiles, batch_spots, device), desc="mini tiles"):
                b, t = X.shape[:2]
                X_subspot = morphology_model(X.to(device, non_blocking=True).flatten(0, 1))
                X_subspot = X_subspot.detach().float().reshape(b, t, -1)

                by_n_neighbors = {}
                for k in range(b):
                    by_n_neighbors.setdefault(len(neighbors[start + k]), []).append(k)
                for n_neighbors, members in by_n_neighbors.items():
                    spots = [start + k for k in members]
                    if n_neighbors:
                        X_neighbors = spot_features[torch.from_numpy(np.stack([neighbors[i] for i in spots]))]
                    elif masks_padding:
                        X_neighbors = torch.zeros(len(spots), 1, spot_features.shape[1])
                    else:
                        X_neighbors = zero_features.expand(len(spots), -1)[:, None]
                    X = [spot_features[spots][:, None].to(device), X_subspot[members], X_neighbors.to(device)]
                    if return_std:
                        expr, std_expr = model_expression(X, return_std=return_std)
                        std_expr = std_expr.detach().cpu().numpy()
                        if std_counts is None:
                            std_counts = np.empty((n_spots, *std_expr.shape[1:]), dtype=std_expr.dtype)
                        std_counts[spots] = std_expr
                    else:
                        expr = model_expression(X)

                    expr = expr.detach().cpu().numpy()
                    expr = model_expression.inverse_transform(expr)
                    if counts is None:
                        counts = np.empty((n_spots, *expr.shape[1:]), dtype=expr.dtype)
                    counts[spots] = expr

                has_nan = np.isnan(counts[start:start + b]).reshape(b, -1).any(axis=1)
                assert not has_nan.any(), obs.iloc[start + int(np.argmax(has_nan))]
                start += b
    # (spots, 1, genes) squeezed, as np.array() of the per-spot (1, genes) outputs was.
    counts = np.expand_dims(counts, 1).squeeze()

    if return_std:
        std_counts = np.expand_dims(std_counts, 1).squeeze()
        return counts, std_counts

    return counts


def predict_cell_spatial_transcriptomics_from_image_path(image_path,
                                                         adata,
                                                         cell_diameter,
                                                         radius_neighbors,
                                                         preprocess,
                                                         morphology_model,
                                                         model_expression,
                                                         device):
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(radius=radius_neighbors)
    neigh.fit(adata.obs[["x_pixel", "y_pixel"]].values)
    neighbors = neigh.radius_neighbors(adata.obs[["x_pixel", "y_pixel"]].values,
                                       return_distance=True, sort_results=True)[1]
    neighbors = [n[1:] for n in neighbors]  # remove the cell itself

    cell_ids = adata.obs.barcode.values

    # Generate neighbors list as strings of formatted cell IDs
    adata.obs["neighbors"] = [
        "___".join(f"{cell_ids[cell_id]}" for cell_id in neigh_ids)
        for neigh_ids in neighbors
    ]

    import pyvips
    image = pyvips.Image.new_from_file(image_path)
    counts = []
    # Set dtype based on the model's precision
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            for _, cell in tqdm(adata.obs.iterrows(), total=len(adata.obs)):

                neighbors_barcodes = cell.neighbors.split('___')

                X_cell = crop_tile(image, cell.x_pixel, cell.y_pixel, cell_diameter)

                X_neighbors = []
                for _, cell_neighbor in adata.obs.query('barcode in @neighbors_barcodes').iterrows():
                    X_neighbors.append(crop_tile(image, cell_neighbor.x_pixel, cell_neighbor.y_pixel, cell_diameter))

                if len(X_neighbors) == 0:
                    X_neighbors = np.zeros((1, *X_cell.shape))

                # Preprocess inputs
                X_cell = preprocess(X_cell).to(device).float()

                X_neighbors = torch.stack([preprocess(x).to(device) for x in X_neighbors]).to(device).float()

                # Apply morphology model to each input
                X_cell = morphology_model(X_cell[None, ])
                X_neighbors = morphology_model(X_neighbors)

                X_cell = detach_and_convert(X_cell)
                X_neighbors = detach_and_convert(X_neighbors)

                X = [X_cell, X_neighbors]
                expr = model_expression(X)
                expr = expr.detach().cpu().numpy()
                expr = model_expression.inverse_transform(expr)
                assert np.isnan(expr).sum() == 0, cell
                counts.append(expr)
    counts = np.array(counts).squeeze()
    return counts
