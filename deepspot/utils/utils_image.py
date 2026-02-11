from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from huggingface_hub import login, hf_hub_download
from transformers import AutoImageProcessor, AutoModel, ViTModel
from collections import OrderedDict
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import json
import timm
import glob
import sys
import os

from .utils_dataloader import compute_neighbors

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
        morphology_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
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
        morphology_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
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
                
            def forward_features(self, x):
                with torch.no_grad():
                    x = self.model(x)
                    return x.last_hidden_state

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


def predict_spot_spatial_transcriptomics_from_image_path(image_path,
                                                         adata,
                                                         spot_diameter,
                                                         n_mini_tiles,
                                                         preprocess,
                                                         morphology_model,
                                                         model_expression,
                                                         device,
                                                         super_resolution=False,
                                                         neighbor_radius=1):
    import pyvips
    image = pyvips.Image.new_from_file(image_path)
    counts = []
    # Set dtype based on the model's precision
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            for _, spot in tqdm(adata.obs.iterrows(), total=len(adata.obs)):
                neighbors_barcodes = compute_neighbors(spot, adata.obs, radius=neighbor_radius)  # "__".join(barcodes)
                neighbors_barcodes = neighbors_barcodes.split('___')

                X_spot = crop_tile(image, spot.x_pixel, spot.y_pixel, spot_diameter)
                X_subspot = compute_mini_tiles(X_spot, n_mini_tiles, super_resolution)
                # X_subspot = np.array(X_subspot).swapaxes(1,3)
                X_neighbors = []
                for _, spot_neighbor in adata.obs.query('barcode in @neighbors_barcodes').iterrows():
                    X_neighbors.append(crop_tile(image, spot_neighbor.x_pixel, spot_neighbor.y_pixel, spot_diameter))

                if len(X_neighbors) == 0:
                    X_neighbors = np.zeros((1, *X_spot.shape))

                # Preprocess inputs
                X_spot = preprocess(X_spot).to(device).float()
                X_subspot = torch.stack([preprocess(x) for x in X_subspot]).to(device).float()
                X_neighbors = torch.stack([preprocess(x).to(device) for x in X_neighbors]).to(device).float()

                # Apply morphology model to each input
                X_spot = morphology_model(X_spot[None, ])
                X_subspot = morphology_model(X_subspot)
                X_neighbors = morphology_model(X_neighbors)

                X_spot = detach_and_convert(X_spot)
                X_subspot = detach_and_convert(X_subspot)
                X_neighbors = detach_and_convert(X_neighbors)

                X = [X_spot, X_subspot, X_neighbors]
                expr = model_expression(X)
                expr = expr.detach().cpu().numpy()
                expr = model_expression.inverse_transform(expr)
                assert np.isnan(expr).sum() == 0, spot
                counts.append(expr)
    counts = np.array(counts).squeeze()
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


def predict_spot2cell_from_image_paths(cell_image_path, 
                                     spot_image_path, 
                                     neighbor_image_paths,
                                     deepspot2cell_model,
                                     morphology_model,
                                     preprocess,
                                     device,
                                     gene_names=None):
    """
    Predict single-cell gene expression using DeepSpot2Cell from individual image paths.
    
    This function is designed for DeepSpot2Cell inference where you have:
    - Individual cell image
    - Spot image containing the cell
    - Neighbor images from the same context
    
    Args:
        cell_image_path (str): Path to the target cell image
        spot_image_path (str): Path to the spot image containing the cell
        neighbor_image_paths (list): List of paths to neighboring images
        deepspot2cell_model: Trained DeepSpot2Cell model
        morphology_model: Morphology foundation model (e.g., PhikonV2)
        preprocess: Preprocessing function for the morphology model
        device (str): Device to run inference on ('cuda' or 'cpu')
        gene_names (list, optional): List of gene names for result mapping
        
    Returns:
        dict: Dictionary mapping gene names/indices to predicted expression values
    """
    from PIL import Image
    import os
    
    def extract_embedding(image_path, morphology_model, preprocess, device):
        """Extract embedding from a single image path."""
        image = Image.open(image_path)
        
        # Ensure square image
        if image.width != image.height:
            size = max(image.width, image.height)
            image = image.resize((size, size))
        
        image_tensor = preprocess(image).to(device).unsqueeze(0)
        
        with torch.no_grad():
            embedding = morphology_model(image_tensor).cpu().numpy()
        
        return embedding

    # Extract cell embedding
    cell_embedding = extract_embedding(cell_image_path, morphology_model, preprocess, device)
    cell_embedding_tensor = torch.tensor(cell_embedding, device=device, dtype=torch.float32)
    
    # Extract spot embedding
    spot_embedding = extract_embedding(spot_image_path, morphology_model, preprocess, device)
    spot_embedding_tensor = torch.tensor(spot_embedding, device=device, dtype=torch.float32)
    
    # Extract neighbor embeddings
    neighbor_embeddings = []
    for neighbor_path in neighbor_image_paths:
        if os.path.exists(neighbor_path):
            neighbor_emb = extract_embedding(neighbor_path, morphology_model, preprocess, device)
            neighbor_embeddings.append(torch.tensor(neighbor_emb, device=device, dtype=torch.float32))
        else:
            # Create zero embedding for missing neighbors
            neighbor_embeddings.append(torch.zeros_like(spot_embedding_tensor))
    
    # Create context (spot + neighbors)
    context_embeddings = [spot_embedding_tensor] + neighbor_embeddings
    context = torch.cat(context_embeddings, dim=0)
    
    # Create context mask (False for valid embeddings, True for missing)
    context_mask = [False]  # Spot is always present
    context_mask.extend([not os.path.exists(path) for path in neighbor_image_paths])
    context_mask = torch.tensor(context_mask, device=device)
    
    # Predict gene expression
    with torch.no_grad():
        predicted_expression = deepspot2cell_model._forward_single_cell(
            cell_embedding_tensor,
            context,
            context_mask
        )
        predicted_expression_np = predicted_expression.cpu().numpy()
    
    # Format results
    results = {}
    if gene_names and len(gene_names) == predicted_expression_np.shape[-1]:
        for i, gene in enumerate(gene_names):
            results[gene] = float(predicted_expression_np[0, i])
    else:
        for i in range(predicted_expression_np.shape[-1]):
            results[f"gene_{i}"] = float(predicted_expression_np[0, i])
    
    return results
