import numpy as np
import h5py
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class DS2CDataset(Dataset):
    def __init__(self, dataset_variant, ids_list, data_path, model_name, standard_scaling=False, shuffle=True, normalize=True, norm_counts=1e4, minmax=False, plot_cell_dist=False, neighb_degree=0, scellst=False, load_cell_types=False):
        self.scellst = scellst
        self.dataset_variant = dataset_variant
        self.neighb_degree = neighb_degree
        self.ids_list = ids_list
        self.data_path = data_path
        self.model_name = model_name
        self.shuffle = shuffle
        self.normalize = normalize
        self.norm_counts = norm_counts
        self.cache = {}
        self.cache_size = 1000
        self.standard_scaling = standard_scaling
        self.spot_scaler = None
        self.cell_scaler = None
        self.minmax = minmax
        self.spot_gene_min = None
        self.spot_gene_max = None
        self.spot_gene_range = None
        self.cell_gene_min = None
        self.cell_gene_max = None
        self.cell_gene_range = None
        self.patch_barcodes = []
        self.load_cell_types = load_cell_types
        self.expression_data = self.load_data()
        self._compute_dataset_statistics(plot_cell_dist=plot_cell_dist)

    def _compute_dataset_statistics(self, plot_cell_dist=False):
        """Compute max cells per patch and min/max normalization values separately for spots and cells"""
        max_cells = 0
        avg_cells = 0
        num_patches = len(self.patch_barcodes)
        cells_per_patch = []
        cells_per_spot = []
        
        if self.minmax:
            first_sample_id = list(self.expression_data.keys())[0]
            first_patch = list(self.expression_data[first_sample_id].keys())[0]
            num_genes = self.expression_data[first_sample_id][first_patch]['spot_expression'].shape[0]

            self.spot_gene_min = np.full(num_genes, np.inf, dtype=np.float32)
            self.spot_gene_max = np.full(num_genes, -np.inf, dtype=np.float32)
            self.cell_gene_min = np.full(num_genes, np.inf, dtype=np.float32)
            self.cell_gene_max = np.full(num_genes, -np.inf, dtype=np.float32)

        if self.standard_scaling and self.normalize:
            all_spot_log_expressions = []
            all_cell_log_expressions = []
        
        for sample_id, patches in self.expression_data.items():
            for patch_barcode, patch_data in patches.items():
                num_cells = len(patch_data['cell_ids'])
                cells_per_patch.append(num_cells)
                if num_cells > max_cells:
                    max_cells = num_cells
                avg_cells += num_cells / num_patches
                
                if self.normalize:
                    spot_expr = patch_data['spot_expression']
                    spot_sum = np.sum(spot_expr)
                    if spot_sum > 0:
                        norm_spot = self.norm_counts * spot_expr / spot_sum
                        log_spot = np.log1p(norm_spot)
                        
                        if self.minmax:
                            self.spot_gene_min = np.minimum(self.spot_gene_min, log_spot)
                            self.spot_gene_max = np.maximum(self.spot_gene_max, log_spot)
                            
                        if self.standard_scaling:
                            all_spot_log_expressions.append(log_spot)
                            
                    if self.normalize:
                        cell_expr = patch_data['cell_expressions']
                        cell_sums = np.sum(cell_expr, axis=1, keepdims=True)
                        cell_ids = patch_data['cell_ids']
                        is_inside_spot = np.array([0.0 if cell_id[-4:] == '_out' else 1.0 
                                                for cell_id in cell_ids], dtype=np.float32)
                        inspot_indices = np.where(is_inside_spot)[0]
                        inspot_total = np.sum(cell_sums[inspot_indices])

                        cells_per_spot.append(len(inspot_indices))
                        
                        if inspot_total > 1e-8:
                            norm_cells = np.zeros_like(cell_expr, dtype=np.float32)
                            norm_cells[inspot_indices] = self.norm_counts * cell_expr[inspot_indices] / inspot_total
                            
                            outspot_indices = np.where(is_inside_spot == 0)[0]
                            if len(outspot_indices) > 0:
                                norm_cells[outspot_indices] = self.norm_counts * cell_expr[outspot_indices] / inspot_total
                            
                            log_cells = np.log1p(norm_cells)
                            nonzero_cells = log_cells[np.sum(cell_expr, axis=1) > 0]
                            
                            if self.minmax and len(nonzero_cells) > 0:
                                self.cell_gene_min = np.minimum(self.cell_gene_min, np.min(nonzero_cells, axis=0))
                                self.cell_gene_max = np.maximum(self.cell_gene_max, np.max(nonzero_cells, axis=0))
                            
                            if self.standard_scaling and len(nonzero_cells) > 0:
                                all_cell_log_expressions.append(nonzero_cells)
        
        # Fit separate scalers for spots and cells
        if self.standard_scaling and self.normalize:
            if all_spot_log_expressions:
                all_spot_log_expressions = np.vstack(all_spot_log_expressions)
                self.spot_scaler = StandardScaler().fit(all_spot_log_expressions)
                print(f"Fit Spot StandardScaler on {all_spot_log_expressions.shape[0]} spot expressions")
                print(f"Spot Mean: min={self.spot_scaler.mean_.min():.4f}, max={self.spot_scaler.mean_.max():.4f}")
                print(f"Spot Std: min={self.spot_scaler.scale_.min():.4f}, max={self.spot_scaler.scale_.max():.4f}")
            
            if all_cell_log_expressions:
                all_cell_log_expressions = np.vstack(all_cell_log_expressions)
                self.cell_scaler = StandardScaler().fit(all_cell_log_expressions)
                print(f"Fit Cell StandardScaler on {all_cell_log_expressions.shape[0]} cell expressions")
                print(f"Cell Mean: min={self.cell_scaler.mean_.min():.4f}, max={self.cell_scaler.mean_.max():.4f}")
                print(f"Cell Std: min={self.cell_scaler.scale_.min():.4f}, max={self.cell_scaler.scale_.max():.4f}")
        
        print(f"Max cells per patch: {max_cells}")
        print(f"Mean cells per patch: {avg_cells}")
        print(f"Mean cells per spot: {np.mean(cells_per_spot):.2f}")
        
        if plot_cell_dist:
            np.save(f'./cells_per_spot_distribution_sample_{self.ids_list[0]}.npy', np.array(cells_per_spot))

        self.max_cells = max_cells
        
        if self.minmax:
            self.spot_gene_range = self.spot_gene_max - self.spot_gene_min
            self.spot_gene_range[self.spot_gene_range < 1e-8] = 1.0
            self.cell_gene_range = self.cell_gene_max - self.cell_gene_min  
            self.cell_gene_range[self.cell_gene_range < 1e-8] = 1.0
            
            print(f"Spot Min values range: {np.min(self.spot_gene_min):.4f} to {np.max(self.spot_gene_min):.4f}")
            print(f"Spot Max values range: {np.min(self.spot_gene_max):.4f} to {np.max(self.spot_gene_max):.4f}")
            print(f"Cell Min values range: {np.min(self.cell_gene_min):.4f} to {np.max(self.cell_gene_min):.4f}")
            print(f"Cell Max values range: {np.min(self.cell_gene_max):.4f} to {np.max(self.cell_gene_max):.4f}")

    def load_data(self):
        """Load all sample data and collect patch barcodes"""
        data = {}
        for sample_id in self.ids_list:
            sample_data = self.load_spot_data(sample_id)
            if sample_data is not None:
                data[sample_id] = sample_data
                for patch_barcode in sample_data.keys():
                    self.patch_barcodes.append((sample_id, patch_barcode))
        
        if self.shuffle:
            np.random.shuffle(self.patch_barcodes)
        
        return data

    def __len__(self):
        return len(self.patch_barcodes)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        attempts = 0
        while True:
            sample_id, patch_barcode = self.patch_barcodes[idx]

            patch_data = self.expression_data[sample_id][patch_barcode]
            cell_expressions = patch_data['cell_expressions']
            cell_ids = patch_data['cell_ids']
            cell_centroids = patch_data['cell_centroids']
            spot_expression = patch_data['spot_expression']
            if self.load_cell_types:
                cell_classes = patch_data['cell_classes']

            spot_sum = np.sum(spot_expression)
            if spot_sum == 0:
                attempts += 1
                if attempts >= 50:
                    raise RuntimeError("Could not find a patch with non-zero spot sum after 50 attempts.")
                idx = np.random.randint(0, len(self.patch_barcodes))
                if idx in self.cache:
                    return self.cache[idx]
                continue

            is_inside_spot = np.array([0.0 if cell_id[-4:] == '_out' else 1.0 for cell_id in cell_ids], dtype=np.float32)

            nonzero_mask = cell_expressions.sum(axis=1) > 0
            if nonzero_mask.any():
                cell_expressions = cell_expressions[nonzero_mask]
                is_inside_spot = is_inside_spot[nonzero_mask]
                if not self.scellst:
                    patch_embeddings = np.load(f"{self.data_path}/embeddings{self.dataset_variant}/{self.model_name}/{patch_barcode}.npy")
                else:
                    patch_embeddings = np.load(f"{self.data_path}/embeddings{self.dataset_variant}/{self.model_name}/{patch_barcode}_scellst.npy")
                patch_embedding = patch_embeddings[0]
                cell_embeddings = patch_embeddings[1:][nonzero_mask]

                if self.normalize:
                    spot_expression = self.preprocess(spot_expression[np.newaxis, ...], is_spot=True)[0]
                    cell_expressions = self.preprocess(cell_expressions, is_spot=False, inspot_mask=is_inside_spot)

                num_cells = len(cell_embeddings)
                emb_dim = cell_embeddings.shape[1] if num_cells else patch_embedding.shape[0]
                expr_dim = cell_expressions.shape[1] if num_cells else spot_expression.shape[0]

                padded_embeds = np.zeros((self.max_cells, emb_dim),  dtype=np.float32)
                padded_exprs = np.zeros((self.max_cells, expr_dim), dtype=np.float32)
                padded_inside = np.zeros(self.max_cells, dtype=np.float32)
                padded_centroids = np.zeros((self.max_cells, 2), dtype=np.float32)
                
                if num_cells:
                    padded_embeds[:num_cells] = cell_embeddings
                    padded_exprs[:num_cells] = cell_expressions
                    padded_inside[:num_cells] = is_inside_spot
                    filtered_centroids = cell_centroids[nonzero_mask]
                    padded_centroids[:num_cells] = filtered_centroids

                cell_mask = np.zeros(self.max_cells, dtype=np.float32)
                cell_mask[:num_cells] = 1.0

                result = {
                    'patch_embedding': torch.tensor(patch_embedding, dtype=torch.float32),
                    'cell_embeddings': torch.tensor(padded_embeds, dtype=torch.float32),
                    'spot_expression': torch.tensor(spot_expression, dtype=torch.float32),
                    'cell_expressions': torch.tensor(padded_exprs, dtype=torch.float32),
                    'cell_mask': torch.tensor(cell_mask, dtype=torch.float32),
                    'is_inside_spot': torch.tensor(padded_inside, dtype=torch.float32),
                    'num_cells': num_cells,
                    'cell_centroids': torch.tensor(padded_centroids, dtype=torch.float32),
                    'patch_barcode': patch_barcode
                }

                if self.neighb_degree:
                    curr_y, curr_x = patch_barcode.split('_')[1:3]
                    curr_y, curr_x = int(curr_y), int(curr_x)
                    neighb_xs = [curr_x + dx for dx in range(-self.neighb_degree, self.neighb_degree + 1)]
                    neighb_ys = [curr_y + dy for dy in range(-self.neighb_degree, self.neighb_degree + 1)]
                    neighb_barcodes = [f"patch_{y}_{x}_{sample_id}" for y in neighb_ys for x in neighb_xs if (y, x) != (curr_y, curr_x)]
                    neighb_embeddings = [patch_embedding.copy()]
                    neighb_masks = [False]
                    for neighb_barcode in neighb_barcodes:
                        if not os.path.exists(f"{self.data_path}/embeddings{self.dataset_variant}/{self.model_name}/{neighb_barcode}.npy"):
                            neighb_embeddings.append(np.zeros(emb_dim, dtype=np.float32))
                            neighb_masks.append(True)
                            continue

                        neighb_emb = np.load(f"{self.data_path}/embeddings{self.dataset_variant}/{self.model_name}/{neighb_barcode}.npy")
                        neighb_embeddings.append(neighb_emb[0])
                        neighb_masks.append(False)
                    
                    neighb_embeddings = np.array(neighb_embeddings, dtype=np.float32)
                    neighb_masks = np.array(neighb_masks, dtype=np.bool_)
                    result['neighb_embeddings'] = torch.tensor(neighb_embeddings, dtype=torch.float32)
                    result['neighb_masks'] = torch.tensor(neighb_masks, dtype=torch.bool)

                if self.load_cell_types:
                    nonzero_cell_classes = cell_classes[nonzero_mask]
                    padded_cell_types = [''] * self.max_cells
                    padded_cell_types[:num_cells] = nonzero_cell_classes.tolist()
                    result['cell_classes'] = padded_cell_types


                if len(self.cache) < self.cache_size:
                    self.cache[idx] = result
                return result
            
            attempts += 1
            if attempts >= 50:
                raise RuntimeError("Could not find a patch with non-zero cells after 50 attempts.")
            idx = np.random.randint(0, len(self.patch_barcodes))
            if idx in self.cache:
                return self.cache[idx]


    def preprocess(self, expression_data, is_spot=False, inspot_mask=None):
        if is_spot:
            summ = np.sum(expression_data, axis=1, keepdims=True)
            norm_data = self.norm_counts * expression_data / summ
        else:
            cell_sums = np.sum(expression_data, axis=1, keepdims=True)
            inspot_indices = np.where(inspot_mask)[0]
            inspot_total = np.sum(cell_sums[inspot_indices])  
            norm_data = np.zeros_like(expression_data, dtype=np.float32) 
            # Normalize cells that are inside the spot by their contribution to total
            norm_data[inspot_indices] = self.norm_counts * expression_data[inspot_indices] / inspot_total
            # Normalize cells that are outside the spot using their own counts (as if they were inside the spot)
            outspot_indices = np.where(inspot_mask == 0)[0]
            if len(outspot_indices) > 0:
                norm_data[outspot_indices] = self.norm_counts * expression_data[outspot_indices] / inspot_total

        log_data = np.log1p(norm_data)

        if self.standard_scaling:
            # Reshape for sklearn compatibility if needed
            orig_shape = log_data.shape
            if len(orig_shape) > 2:
                log_data = log_data.reshape(-1, orig_shape[-1])

            if is_spot and self.spot_scaler is not None:
                log_data = self.spot_scaler.transform(log_data)
            elif not is_spot and self.cell_scaler is not None:
                log_data = self.cell_scaler.transform(log_data)
        
            # Reshape back if needed
            if len(orig_shape) > 2:
                log_data = log_data.reshape(orig_shape)

        if self.minmax:
            if is_spot:
                log_data = (log_data - self.spot_gene_min) / self.spot_gene_range
            else:
                log_data = (log_data - self.cell_gene_min) / self.cell_gene_range
            
        return log_data
    
    def inverse_transform(self, scaled_data, is_spot=False):
        """Inverse transform standardized and/or min-max normalized data back to log space
        """
        orig_shape = scaled_data.shape
        if isinstance(scaled_data, torch.Tensor):
            scaled_data = scaled_data.cpu().numpy()
            
        if len(orig_shape) > 2:
            scaled_data = scaled_data.reshape(-1, orig_shape[-1])
        
        data = scaled_data.copy()
        if self.minmax:
            if is_spot:
                data = data * self.spot_gene_range + self.spot_gene_min
            else:
                data = data * self.cell_gene_range + self.cell_gene_min
        
        if self.standard_scaling:
            if is_spot and self.spot_scaler is not None:
                data = self.spot_scaler.inverse_transform(data)
            elif not is_spot and self.cell_scaler is not None:
                data = self.cell_scaler.inverse_transform(data)
        
        # Reshape back if needed
        if len(orig_shape) > 2:
            data = data.reshape(orig_shape)
            
        return data


    def spatial_upsample_and_smooth(self, genex_data, cell_expressions):
        # Not used for now
        pass


    def load_spot_data(self, sample_id, patch_barcode=None):
        try:
            with h5py.File(f'{self.data_path}/expressions{self.dataset_variant}/{sample_id}_expressions.h5', 'r') as f:
                if patch_barcode is not None:
                    # Load specific patch
                    if patch_barcode in f:
                        spot_group = f[patch_barcode]
                        data = {
                            'spot_expression': spot_group['spot_expression'][:],
                            'cell_expressions': spot_group['cell_expressions'][:],
                            'cell_ids': [id.decode('utf-8') for id in spot_group['cell_ids'][:]], 
                            'cell_centroids': spot_group['cell_centroids'][:]
                        }
                        if self.load_cell_types and 'cell_classes' in spot_group:
                            data['cell_classes'] = np.array([s.decode('utf-8') for s in spot_group['cell_classes'][:]])
                        return data
                    else:
                        return None
                else:
                    # Load all patches
                    result = {}
                    for pb in f.keys():
                        data = {
                            'spot_expression': f[pb]['spot_expression'][:],
                            'cell_expressions': f[pb]['cell_expressions'][:],
                            'cell_ids': [id.decode('utf-8') for id in f[pb]['cell_ids'][:]], 
                            'cell_centroids': f[pb]['cell_centroids'][:]
                        }
                        if self.load_cell_types and 'cell_classes' in f[pb]:
                            data['cell_classes'] = np.array([s.decode('utf-8') for s in f[pb]['cell_classes'][:]])
                        result[pb] = data
                    return result
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading data for sample {sample_id}: {e}")
            return None