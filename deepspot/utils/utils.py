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
import yaml
import json
from hest import iter_hest


def model_fine_tune(model, dataloader, rho=False, gene_expression=True, max_epochs=10):
    # Freeze all layers in the model
    for param in model.parameters():
        param.requires_grad = False

    if rho:
        # Unfreeze the 'gene_expression' layer
        for param in model.rho.parameters():
            param.requires_grad = True
        # Unfreeze the 'gene_expression' layer
    if gene_expression:
        for param in model.gene_expression.parameters():
            param.requires_grad = True

    # Step 2: Update the optimizer to include only the trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=model.hparams.lr, weight_decay=model.hparams.weight_decay
    )
    model.eval()

    # Set 'rho' and 'gene_expression' layers to train mode (since they are fine-tuned)
    if rho:
        model.rho.train()
    if gene_expression:
        model.gene_expression.train()
    # Step 3: Train the model using the fine-tuning DataLoader
    trainer = L.Trainer(max_epochs=max_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)


def run_inference_from_dataloader(model, dataloader, device):
    model.to(device)  # same device
    model.eval()

    out = []

    with torch.no_grad():
        for X, _ in tqdm(dataloader):
            if type(X) is list:
                X = (x.to(device) for x in X)
            else:
                X = X.to(device)
            y = model.forward(X)
            # y_zeros = y_zeros.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            out.extend(y)

    return np.array(out)


def plot_loss_values(train_losses, val_losses=None):
    train_losses = np.array(train_losses)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    train_idx = np.arange(0, len(train_losses))
    plt.plot(train_idx, train_losses, color="b", label="train")

    if val_losses is not None:
        val_losses = np.array(val_losses)
        val_idx = np.arange(0, len(val_losses)) * (len(train_losses) // len(val_losses) + 1)
        plt.plot(val_idx, val_losses, color="r", label="val")

    plt.legend()


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def order_genes(config):
    """Order genes by variability across samples."""
    data_folder = config['data']['data_folder']
    nc_min = config.get('min_cells', 20)
    qv_min = config.get('qv_thr', 20)
    dataset = config['data']['dataset']
    sample_ids = config['data'][dataset]['train']['dataset_ids']

    all_genes = set()
    all_sample_filtered_genes = []
    all_cell_expressions = []
    gene_to_idx_map = {}
    
    for i, st in enumerate(iter_hest(data_folder, id_list=sample_ids, load_transcripts=True)):
        sample_id = sample_ids[i]
        print(f"    Sample {i+1}/{len(sample_ids)}: {sample_id}")
        
        adata = st.adata
        all_genes.update(adata.var_names)
        
        transcript_df = st.transcript_df
        if transcript_df is None or transcript_df.empty:
            print(f"      No transcript data available. Skipping.")
            all_sample_filtered_genes.append(set())
            continue
            
        print(f"      Filtering transcripts: QV > {qv_min}")
        transcript_df['qv'] = pd.to_numeric(transcript_df['qv'], errors='coerce')
        df_qv_filtered = transcript_df[transcript_df['qv'] > qv_min]
        
        df_cell_filtered = df_qv_filtered[
            ~df_qv_filtered['cell_id'].isin(['UNASSIGNED', -1]) & 
            df_qv_filtered['cell_id'].notna()
        ]
        
        if df_cell_filtered.empty:
            print(f"      No valid transcripts after filtering. Skipping.")
            all_sample_filtered_genes.append(set())
            continue
            
        # Count cells per gene
        gene_cell_counts = df_cell_filtered.groupby('feature_name')['cell_id'].nunique()
        filtered_genes = set(gene_cell_counts[gene_cell_counts >= nc_min].index)
        all_sample_filtered_genes.append(filtered_genes)
        
        print(f"      Found {len(filtered_genes)} genes present in ≥{nc_min} cells with QV>{qv_min}")
        
        for cell_id, cell_data in df_cell_filtered.groupby('cell_id'):
            gene_counts = cell_data.groupby('feature_name').size().to_dict()
            all_cell_expressions.append((gene_counts, sample_id))
    
    if all_sample_filtered_genes:
        quality_filtered_genes = set.intersection(*all_sample_filtered_genes)
        print(f"\n    Found {len(quality_filtered_genes)} genes passing quality filters in ALL samples")
    else:
        quality_filtered_genes = set()
        print("\n    No genes passed quality filters across all samples")
    
    print("    Calculating gene variability...")
    gene_list = sorted(list(quality_filtered_genes))
    gene_to_idx_map = {gene: idx for idx, gene in enumerate(gene_list)}
    expression_matrix = []
    
    for gene_counts, sample_id in all_cell_expressions:
        gene_expression = np.zeros(len(gene_list))
        for gene_name, count in gene_counts.items():
            if gene_name in gene_to_idx_map:
                gene_expression[gene_to_idx_map[gene_name]] = count
        
        expression_matrix.append(gene_expression)
    
    if not expression_matrix:
        print("    No expression data available after filtering. Cannot order genes.")
        return []

    print(f"    Creating AnnData with {len(expression_matrix)} cells and {len(gene_list)} genes")
    combined_adata = ad.AnnData(X=np.vstack(expression_matrix), var=pd.DataFrame(index=gene_list))

    print("    Calculating gene variability...")
    sc.pp.highly_variable_genes(combined_adata, flavor='seurat_v3')
    idx = combined_adata.var['highly_variable_rank'].argsort()
    combined_adata = combined_adata[:, idx]
    ordered_gene_names = combined_adata.var_names.tolist()
    
    ordered_genes_path = f"{data_folder}/{config['data'][dataset]['ordered_genes_file']}"
    with open(ordered_genes_path, 'w') as f:
        json.dump(ordered_gene_names, f, indent=2)
    print(f"    Saving ordered genes to {ordered_genes_path}")
