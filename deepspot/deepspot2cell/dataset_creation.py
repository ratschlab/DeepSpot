import os
import gc
import json
import h5py
import pyvips
import pandas as pd
from shapely.geometry import box, Polygon, shape, Point
import argparse
from shapely.ops import unary_union
import numpy as np
from hest import iter_hest
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import time

from deepspot.utils.utils import load_config, order_genes
from deepspot.utils.utils_image import format_to_dtype, get_morphology_model_and_preprocess


def process_batch(batch_patches, model, preprocess, device):
    batch_processed = []
    for patch in batch_patches:
        patch_img = Image.fromarray(patch)
        if patch_img.width != patch_img.height:
            size = max(patch_img.width, patch_img.height)
            patch_img = patch_img.resize((size, size))
        batch_processed.append(preprocess(patch_img).to(device).to(torch.float16))
    
    batch_tensor = torch.stack(batch_processed)
    
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            batch_embeddings = model(batch_tensor)
            batch_embeddings = batch_embeddings.detach().cpu()
    
    return batch_embeddings


def get_spot_embs(X_spot, morphology_model, preprocess, device, dtype_to_use=torch.float16):
    with torch.autocast(device_type="cuda", dtype=dtype_to_use):
        with torch.inference_mode():
            X_spot_img = Image.fromarray(X_spot)
            # if spot is not square, make it square
            if X_spot_img.width != X_spot_img.height:
                size = max(X_spot_img.width, X_spot_img.height)
                X_spot_img = X_spot_img.resize((size, size))

            X_spot_tensor = preprocess(X_spot_img).to(device).to(torch.float16)
            token_map = morphology_model(X_spot_tensor[None, ])
            cls_tok = token_map.squeeze(0)

        return cls_tok.cpu().numpy()


def sort_genes_by_variability(adata, config):
    ordered_genes_path = f"{config['data']['data_folder']}/{config['data'][config['dataset']]['ordered_genes_file']}"
    print(f"  Using gene order from {ordered_genes_path}")
    
    if os.path.exists(ordered_genes_path):
        print(f"  Loading ordered genes from {ordered_genes_path}")
    else:
        print(f"  Ordered genes file not found. Generating new order.")
        order_genes(config)

    with open(ordered_genes_path, 'r') as f:
        ordered_gene_names = json.load(f)


    adata_genes_set = set(adata.var.index.values)
    ordered_genes_set = set(ordered_gene_names)

    if adata_genes_set != ordered_genes_set:
        print("  Warning: Gene names in adata and ordered genes file do not match.")
        diff = adata_genes_set - ordered_genes_set
        if len(diff) > 0:
            print(f"  Different genes in adata: {diff}")

    return ordered_gene_names


def crop_tile(image, x_pixel, y_pixel, spot_diameter):
    spot = image.crop(x_pixel, y_pixel, spot_diameter, spot_diameter)
    main_tile = np.ndarray(buffer=spot.write_to_memory(),
                           dtype=format_to_dtype[spot.format],
                           shape=[spot.height, spot.width, spot.bands])
    main_tile = main_tile[:, :, :3]
    return main_tile
        

def fix_invalid_geometry(geom):
    """Attempt to fix invalid geometry"""
    if not geom.is_valid:
        try:
            fixed = geom.buffer(0)
            if fixed.is_valid:
                return fixed

            if hasattr(geom, 'make_valid'):
                fixed = geom.make_valid()
                if fixed.is_valid:
                    return fixed
        except Exception:
            pass
    return geom


def create_tissue_mask(tissue_mask_json, sample_id):
    """Create a valid tissue mask from GeoJSON, handling geometry errors robustly"""
    tissue_polygons = []
    total_polygons = 0
    invalid_polygons = 0
    
    for feature in tissue_mask_json['features']:
        if feature['geometry']['type'] != 'Polygon':
            continue
        total_polygons += 1

        try:
            poly = shape(feature['geometry'])
            if not poly.is_valid:
                poly = poly.buffer(0)
                if not poly.is_valid:
                    invalid_polygons += 1
                    continue
            tissue_polygons.append(poly)         
        except Exception:
            invalid_polygons += 1
    
    if not tissue_polygons:
        print(f"  Warning: No valid polygons in {sample_id} (total: {total_polygons}, invalid: {invalid_polygons})")
        return None
    print(f"  Extracted {len(tissue_polygons)} valid polygons from {total_polygons} total")
    if len(tissue_polygons) == 1:
        return tissue_polygons[0]

    try:
        return unary_union(tissue_polygons)
    except Exception:
        pass

    mask = tissue_polygons[0]
    successful_unions = 1
    
    for i, poly in enumerate(tissue_polygons[1:], 1):
        try:
            clean_mask = mask.buffer(0)
            clean_poly = poly.buffer(0)
            if clean_mask.is_valid and clean_poly.is_valid:
                mask = clean_mask.union(clean_poly)
                successful_unions += 1
            else:
                print(f"  Warning: Skipping polygon {i} in union due to validity issues")
        except Exception:
            print(f"  Warning: Failed to union polygon {i}, continuing with current mask")
    print(f"  Created tissue mask with {successful_unions}/{len(tissue_polygons)} polygons")
    return mask


def process_sample(sample_id, st, model, preprocess, device, batch_size, config):
    data_folder = config['data']['data_folder']
    dataset_variant = config['data']['dataset_variant']
    print(f"Processing {sample_id}")
    if os.path.exists(f'{data_folder}/expressions{dataset_variant}/{sample_id}_expressions.h5'):
        print(f"  Already processed {sample_id}, skipping")
        return
    
    with open(f"{data_folder}/metadata/{sample_id}.json") as f:
        metadata = json.load(f)
    print(metadata)
    um_per_px = metadata.get('pixel_size_um_estimated', 0.2125)
    if config['data'].get('spot_diameter_px', None) is not None:
        spot_diameter = config['data']['spot_diameter_px']
        patch_size = config['data']['patch_size_px']
    else:
        spot_diameter = int(config['data']['spot_diameter'] / um_per_px)
        patch_size = int(config['data']['patch_size'] / um_per_px)
    print(f"  Pixel size (um): {um_per_px}, spot diameter (px): {spot_diameter}, patch size (px): {patch_size}")

    ordered_gene_names = sort_genes_by_variability(st.adata, config)
    gene_to_idx = {gene: idx for idx, gene in enumerate(ordered_gene_names)}

    with open(f"{data_folder}/tissue_seg/{sample_id}_contours.geojson", 'r') as f:
        tissue_mask_json = json.load(f)
    tissue_mask = create_tissue_mask(tissue_mask_json, sample_id)

    #patches = h5py.File(f"{data_folder}/patches/{sample_id}.h5")
    cell_shapes = st.get_shapes('xenium_cell', 'he').shapes
    if cell_shapes.index.dtype == 'object':
        decode = lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)
        cell_shapes.index = cell_shapes.index.map(decode).astype("string")
    #print(f'Cell shapes index data type: {cell_shapes.index.dtype}')


    wsi = pyvips.Image.new_from_file(f"{data_folder}/wsis/{sample_id}.tif")
    wsi_width = wsi.width
    wsi_height = wsi.height
    num_patches_h = wsi_height // patch_size
    num_patches_w = wsi_width // patch_size
    print(f"  Divided wsi into {num_patches_h} x {num_patches_w} = {num_patches_h * num_patches_w} patches")

    # filtering ---------------------------------------
    filtered_df = st.transcript_df[~st.transcript_df.cell_id.isin(['UNASSIGNED', -1])]
    
    qv_thr = config['data']['qv_thr']
    initial_transcript_count = len(filtered_df)
    filtered_df['qv'] = pd.to_numeric(filtered_df['qv'], errors='coerce')
    filtered_df = filtered_df[filtered_df['qv'] > qv_thr]
    print(f"  Filtered by 'qv' > {qv_thr}: {len(filtered_df)} transcripts retained from {initial_transcript_count}.")
    # /filtering --------------------------------------
    #print(filtered_df.head())
    #print(cell_shapes.head())
    for col in ('feature_name', 'cell_id'):
        s = filtered_df[col]
        if s.dtype == 'object':
            # decode *only* the bytes, leave real strings untouched
            decoded = s.str.decode('utf-8', errors='ignore')
            filtered_df[col] = decoded.fillna(s).astype('string')
    #print(filtered_df.head())
    #print(f'filtered df cell_id data type: {filtered_df["cell_id"].dtype}')

    cell_dict = {}
    cell_batch = []
    cell_batch_ids = []
    approx_total_cells = metadata.get('cells_under_tissue', 200000) * 1.5
    total_cells = 0
    cells_inside_tissue = 0

    cell_start_time = time.time()
    cell_count = 0
    print(f"  Starting processing of cells for {sample_id}")

    for cell_id, cell_data in filtered_df.groupby('cell_id'):
        cell_count += 1
        total_cells += 1
        if cell_count % 10000 == 0:
            elapsed = time.time() - cell_start_time
            cells_per_sec = cell_count / elapsed
            remaining = (approx_total_cells - cell_count) / cells_per_sec if cells_per_sec > 0 else 0
            print(f"  Processed {cell_count}/{approx_total_cells} cells ({cell_count/approx_total_cells*100:.1f}%) - "
                  f"ETA: {int(elapsed/3600)}:{int(elapsed/60)}:{int(elapsed)%60}/{int(remaining/3600)}:{int(remaining/60)}:{int(remaining)%60}")
        
        gene_expression = np.zeros(len(ordered_gene_names))
        gene_counts = cell_data.groupby('feature_name').size()
        for gene_name, count in gene_counts.items():
            if gene_name in gene_to_idx:
                gene_expression[gene_to_idx[gene_name]] = count
        if np.sum(gene_expression) == 0:
            #print('zero ex')
            continue
        
        cell_shape_data = cell_shapes[cell_shapes.index == cell_id]
        if cell_shape_data.empty:
            #print('empty shape')
            continue
            
        cell_shape = cell_shape_data.geometry.values[0]  # shapely polygon
        if not cell_shape.is_valid:
            cell_shape = fix_invalid_geometry(cell_shape)
            if not cell_shape.is_valid:
                print(f"  Cell {cell_id} has invalid geometry after fixing")
                continue
        centroid = cell_shape.centroid
        cell_centroid_x = centroid.x
        cell_centroid_y = centroid.y
        
        # Check if cell is inside tissue mask
        if not tissue_mask.contains(Point(cell_centroid_x, cell_centroid_y)):
            #print('actual outside')
            continue
        
        cells_inside_tissue += 1
        cell_area = cell_shape.area
        cell_bounds = cell_shape.bounds  # (minx, miny, maxx, maxy)
        # Calculate the smallest square bounding box
        rect_width = cell_bounds[2] - cell_bounds[0]
        rect_height = cell_bounds[3] - cell_bounds[1]
        max_dimension = max(rect_width, rect_height)
        square_half_side = max_dimension / 2
        square_minx = cell_centroid_x - square_half_side
        square_miny = cell_centroid_y - square_half_side
        square_maxx = cell_centroid_x + square_half_side
        square_maxy = cell_centroid_y + square_half_side

        try:
            cell_img = crop_tile(wsi, square_minx, square_miny, max_dimension)
        except Exception as e:
            print(f"  Error cropping cell {cell_id} image: {e}")
            continue

        cell_dict[f"{sample_id}_{cell_id}"] = {
            'cell_expression': gene_expression,
            'cell_shape': cell_shape,
            'cell_centroid_x': cell_centroid_x,
            'cell_centroid_y': cell_centroid_y,
            'area': cell_area,
            'cell_bounds': cell_bounds,
            'square_bbox_coords': (square_minx, square_miny, square_maxx, square_maxy),
        }

        cell_batch.append(cell_img)
        cell_batch_ids.append(f"{sample_id}_{cell_id}")
        if len(cell_batch) == batch_size:
            cell_embeddings = process_batch(cell_batch, model, preprocess, device)
            # if scellst_model is not None:
            #     scellst_embeddings = process_batch(cell_batch, scellst_model, scellst_preprocess, device)
            for j, c_id in enumerate(cell_batch_ids):
                cell_dict[c_id]['cell_embedding'] = cell_embeddings[j]
                #cell_dict[c_id]['scellst_embedding'] = scellst_embeddings[j] if scellst_model else None
            cell_batch = []
            cell_batch_ids = []

    if len(cell_batch) > 0:
        cell_embeddings = process_batch(cell_batch, model, preprocess, device)
        # if scellst_model is not None:
        #     scellst_embeddings = process_batch(cell_batch, scellst_model, scellst_preprocess, device)
        for j, c_id in enumerate(cell_batch_ids):
            cell_dict[c_id]['cell_embedding'] = cell_embeddings[j]
            #cell_dict[c_id]['scellst_embedding'] = scellst_embeddings[j] if scellst_model else None

    elapsed = time.time() - cell_start_time
    print(f"  Completed cell processing: {cells_inside_tissue}/{total_cells} cells inside tissue in {elapsed:.1f} sec")

    print(f"  Created cell data for {cells_inside_tissue} cells inside tissue ({cells_inside_tissue}/{total_cells} cells are within tissue)")

    patches_inside_tissue = 0
    total_patches = num_patches_h * num_patches_w
    spot_dict = {}

    patch_start_time = time.time()
    patch_count = 0
    total_patch_count = num_patches_h * num_patches_w
    print(f"  Starting processing of patches")

    for h_idx in range(num_patches_h):
        for w_idx in range(num_patches_w):
            patch_count += 1
            # Print progress every 1k patches
            if patch_count % 1000 == 0:
                elapsed = time.time() - patch_start_time
                patches_per_sec = patch_count / elapsed
                remaining = (total_patch_count - patch_count) / patches_per_sec if patches_per_sec > 0 else 0
                print(f"  Processed {patch_count}/{total_patch_count} patches ({patch_count/total_patch_count*100:.1f}%) - "
                      f"ETA: {int(elapsed/3600)}:{int(elapsed/60)}:{int(elapsed)%60}/{int(remaining/3600)}:{int(remaining/60)}:{int(remaining)%60}")

            patch_barcode = f"patch_{h_idx}_{w_idx}_{sample_id}"
            patch_y = h_idx * patch_size
            patch_x = w_idx * patch_size
            patch_center_x = patch_x + patch_size // 2
            patch_center_y = patch_y + patch_size // 2
            
            # Skip patches whose centroid is outside tissue
            if not tissue_mask.contains(Point(patch_center_x, patch_center_y)):
                continue
                
            patches_inside_tissue += 1
            search_radius = patch_size
            nearby_cells = {
                cell_id: cell_data for cell_id, cell_data in cell_dict.items()
                if (abs(cell_data['cell_centroid_x'] - patch_center_x) <= search_radius and 
                    abs(cell_data['cell_centroid_y'] - patch_center_y) <= search_radius)
            }
            if len(nearby_cells) == 0:
                continue
            
            #print(f"  Processing patch {patch_barcode} with {len(nearby_cells)} nearby cells")
            patch_img = crop_tile(wsi, patch_x, patch_y, patch_size)
            patch_box = box(patch_x, patch_y, patch_x + patch_size, patch_y + patch_size)
            cell_expressions = []
            cell_bboxes_adjusted = []
            cell_centroids_adjusted = []

            cell_ids = []
            cell_ids_tagged = []

            if not patch_box.is_valid:
                patch_box = fix_invalid_geometry(patch_box)
                if not patch_box.is_valid:
                    print(f"  Patch {patch_barcode} has invalid geometry after fixing")
                    continue

            spot_radius = spot_diameter // 2
            spot = Point(patch_center_x, patch_center_y).buffer(spot_radius, cap_style=1)    
            
            for cell_id, cell_data in nearby_cells.items():
                cell_shape = cell_data['cell_shape']
                cell_bounds = cell_data['cell_bounds']
                patch_bounds = patch_box.bounds  # (minx, miny, maxx, maxy)

                # If theres no intersection at all skip
                if (cell_bounds[2] < patch_bounds[0] or cell_bounds[0] > patch_bounds[2] or
                    cell_bounds[3] < patch_bounds[1] or cell_bounds[1] > patch_bounds[3]):
                    continue

                # If cell is fully contained in the patch accept it
                if (cell_bounds[0] >= patch_bounds[0] and cell_bounds[2] <= patch_bounds[2] and
                    cell_bounds[1] >= patch_bounds[1] and cell_bounds[3] <= patch_bounds[3]):
                    #intersection = cell_shape
                    pass
                else:
                    if cell_shape.intersects(patch_box):
                        intersection = cell_shape.intersection(patch_box)
                        if intersection.area / cell_data['area'] < 0.6:
                            continue
                    else:
                        continue

                # chek if cell is inside the spot
                intersection_area = cell_shape.intersection(spot).area
                if intersection_area / cell_shape.area >= 0.60:
                    cell_id_tagged = f"{cell_id}_in"
                else:
                    cell_id_tagged = f"{cell_id}_out"
                
                cell_expressions.append(cell_data['cell_expression'])
                cell_ids.append(cell_id)
                cell_ids_tagged.append(cell_id_tagged)
                cell_bbox_coords = cell_data['square_bbox_coords']
                # adjust and normalize the bounding box and centroid coordinates to the patch size
                cell_bbox_adjusted_normalized = (
                    (cell_bbox_coords[0] - patch_x) / patch_size,  # normalized minx
                    (cell_bbox_coords[1] - patch_y) / patch_size,  # normalized miny
                    (cell_bbox_coords[2] - patch_x) / patch_size,  # normalized maxx
                    (cell_bbox_coords[3] - patch_y) / patch_size   # normalized maxy
                )
                cell_bboxes_adjusted.append(cell_bbox_adjusted_normalized)
                cell_centroid_x = (cell_data['cell_centroid_x'] - patch_x) / patch_size
                cell_centroid_y = (cell_data['cell_centroid_y'] - patch_y) / patch_size
                cell_centroids_adjusted.append((cell_centroid_x, cell_centroid_y))

            if len(cell_expressions) == 0:
                continue

            cell_expressions = np.array(cell_expressions)
            spot_expression = np.zeros(len(ordered_gene_names))
            for i, cell_id in enumerate(cell_ids_tagged):
                if '_in' in cell_id:
                    spot_expression += cell_expressions[i]


            spot_dict[patch_barcode] = {
                #'patch_img': patch_img,
                'spot_expression': spot_expression,
                #'patch_bounds': patch_box.bounds,  # (minx, miny, maxx, maxy)
                'cell_ids': cell_ids_tagged,
                'cell_expressions': cell_expressions,
                'cell_bboxes_adjusted': cell_bboxes_adjusted,
                'cell_centroids_adjusted': cell_centroids_adjusted,
            }

            spot_embeddings = get_spot_embs(patch_img, model, preprocess, device)

            # Save patch and cell embeddings
            patch_embeddings = [spot_embeddings.astype(np.float16)]
            for cell_id in cell_ids:
                emb = cell_dict[cell_id]['cell_embedding']
                if torch.is_tensor(emb):
                    emb = emb.numpy()
                patch_embeddings.append(emb.astype(np.float16))
                del cell_dict[cell_id]['cell_embedding']
                cell_dict[cell_id]['cell_embedding'] = None
            patch_embeddings = np.vstack(patch_embeddings)
            np.save(f'{data_folder}/embeddings{dataset_variant}/{model_name}/{patch_barcode}.npy', patch_embeddings)


    elapsed = time.time() - patch_start_time
    print(f"  Completed patch processing: {patches_inside_tissue}/{total_patch_count} patches inside tissue in {elapsed:.1f} sec")

    print(f"  Saving data for {len(spot_dict)} spots to HDF5 file...")
    with h5py.File(f'{data_folder}/expressions{dataset_variant}/{sample_id}_expressions.h5', 'w') as f:
        for patch_barcode, patch_data in spot_dict.items():
            spot_group = f.create_group(patch_barcode)
            spot_group.attrs['spot_diameter'] = spot_diameter
            spot_group.attrs['patch_size'] = patch_size
            spot_group.attrs['um_per_px'] = um_per_px
            spot_group.create_dataset('spot_expression', data=patch_data['spot_expression'], compression='gzip', compression_opts=4)
            spot_group.create_dataset('cell_expressions', data=patch_data['cell_expressions'], compression='gzip', compression_opts=4)
            cell_ids = np.array(patch_data['cell_ids'], dtype=h5py.string_dtype())
            spot_group.create_dataset('cell_ids', data=cell_ids)
            spot_group.create_dataset('cell_centroids', data=np.array(patch_data['cell_centroids_adjusted'], dtype=np.float32), compression='gzip', compression_opts=4)
            spot_group.create_dataset('cell_bbxs', data=np.array(patch_data['cell_bboxes_adjusted'], dtype=np.float32), compression='gzip', compression_opts=4)
    print(f"  Saved spot data to {data_folder}/expressions{dataset_variant}/{sample_id}_expressions.h5")
    print(f"  Processed {patches_inside_tissue} patches inside tissue ({patches_inside_tissue}/{total_patches} patches are within tissue)")

    del spot_dict, cell_dict
    gc.collect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process slides for patch embeddings')
    parser.add_argument('--config', type=str, default='experiments/config.yaml', help='Path to config file (default: experiments/config.yaml)')
    parser.add_argument('--indices', type=int, nargs='+', help='List of specific indices to process')
    parser.add_argument('--dataset', type=str, default='lung', help='Dataset to process (default: lung)')
    args = parser.parse_args()

    sample_ids = []
    config = load_config(args.config)
    config['dataset'] = args.dataset
    data_folder = config['data']['data_folder']
    model_name = config['data']['model_name']
    model_path = config['data']['model_path']
    dataset_variant = config['data']['dataset_variant']
    sample_ids = config['data'][args.dataset]['train']['dataset_ids'] + config['data'][args.dataset]['ood']['dataset_ids']
    if args.indices:
        sample_ids = [sample_ids[i] for i in args.indices]

    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)
    batch_size = int(vram_gb * 32 / 14) if vram_gb > 13 else 16
    print(f"Auto-configured batch size to {batch_size} based on {vram_gb:.1f}GB VRAM")

    model, preprocess, feature_dim = get_morphology_model_and_preprocess(model_name, 'cuda', model_path=model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    os.makedirs(f'{data_folder}/embeddings{dataset_variant}/', exist_ok=True)
    os.makedirs(f'{data_folder}/embeddings{dataset_variant}/{model_name}', exist_ok=True)
    os.makedirs(f'{data_folder}/expressions{dataset_variant}', exist_ok=True)

    for i, st in enumerate(iter_hest(data_folder, id_list=sample_ids, load_transcripts=True)):
        sample_id = sample_ids[i]
        process_sample(sample_id, st, model, preprocess, device, batch_size, config)