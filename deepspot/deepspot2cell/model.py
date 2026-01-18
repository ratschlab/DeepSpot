from torch import nn
import torch
import numpy as np
import lightning as L
from typing import Union, Dict

from deepspot.deepspot2cell.loss import mse, safe_pearson
from deepspot.utils.utils import fix_seed


class Phi(nn.Module):
    def __init__(self, input_size: int, output_size: int, p: float = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Dropout(p=p),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
            nn.Dropout(p=p),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Rho(nn.Module):
    def __init__(self, input_size: int, output_size: int, p: float = 0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, output_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class DeepSpot2Cell(L.LightningModule):
    def __init__(self, input_size: int, output_size: int, lr: float = 1e-4, 
                 p: float = 0.3, p_phi: Union[None, float] = None,
                 p_rho: Union[None, float] = None, n_ensemble: int = 10,
                 n_ensemble_phi: Union[None, int] = None,
                 n_ensemble_rho: Union[None, int] = None, phi2rho_size: int = 512,
                 weight_decay: float = 1e-6, random_seed: int = 2024, cell_gt_available = True):
        super().__init__()

        self.save_hyperparameters()
        fix_seed(random_seed)

        self.model_name = f"deepspot2cell"
        self.p_phi = p_phi or p
        self.p_rho = p_rho or p
        self.n_ensemble_phi = n_ensemble_phi or n_ensemble
        self.n_ensemble_rho = n_ensemble_rho or n_ensemble
        self.loss_fn = mse
        
        self.train_batch_losses = []
        self.train_cell_inside_preds = []
        self.train_cell_inside_targets = []
        self.train_cell_outside_preds = []
        self.train_cell_outside_targets = []
        
        self.val_batch_losses = []
        self.val_cell_inside_preds = []
        self.val_cell_inside_targets = []
        self.val_cell_outside_preds = []
        self.val_cell_outside_targets = []

        mult = 3
        self.phi_spot = nn.ModuleList([Phi(input_size, phi2rho_size, self.p_phi)
                                       for _ in range(self.n_ensemble_phi)])
        self.phi_cell = nn.ModuleList([Phi(input_size, phi2rho_size, self.p_phi)
                                       for _ in range(self.n_ensemble_phi)])
        self.rho = nn.ModuleList([Rho(phi2rho_size * mult, output_size, self.p_rho)
                                  for _ in range(self.n_ensemble_rho)])
        
        self.cell_gt_available = cell_gt_available

    def forward(self, x, mask, context, context_mask):  
        return self._forward_superspot(x, mask, context, context_mask)


    def _forward_single_cell(self, x, context, context_mask):
        x = x.unsqueeze(0) if x.dim() == 1 else x
        n_cells, d_cell_emb = x.shape
        phi_cells = self._apply_phi_cell(x)

        phi_context_spot_own = self._apply_phi(context[0].unsqueeze(0))

        neighbor_contexts = context[1:, :] 
        neighbor_masks = context_mask[1:]
        d_ctx_emb = neighbor_contexts.shape[-1]
        phi_neighbor_contexts = self._apply_phi(
                neighbor_contexts.reshape(-1, d_ctx_emb)
            ).view(neighbor_contexts.shape[0], self.hparams.phi2rho_size)
        phi_neighbor_contexts_masked_for_sum = phi_neighbor_contexts.masked_fill(
                neighbor_masks.unsqueeze(-1), 0.0
            )
        sum_phi_neighbor_contexts = phi_neighbor_contexts_masked_for_sum.sum(dim=0, keepdim=True)
        num_valid_neighbors = (~neighbor_masks).sum(dim=0, keepdim=True).float().clamp(min=1)
        mean_phi_neighbor_contexts = sum_phi_neighbor_contexts / num_valid_neighbors

        phi_context_spot_own_expanded = phi_context_spot_own.expand(n_cells, -1)
        mean_phi_neighbor_contexts_expanded = mean_phi_neighbor_contexts.expand(n_cells, -1)

        concatenated_features = torch.cat([phi_cells, phi_context_spot_own_expanded, mean_phi_neighbor_contexts_expanded], dim=1)
        return self._apply_rho(concatenated_features)
    
    def _forward_superspot(self, x, mask=None, context=None, context_mask=None):
        b, c, d_cell_emb = x.shape
        _b_ctx, n_total_ctx, d_ctx_emb = context.shape

        phi_cells = self._apply_phi_cell(x.view(-1, d_cell_emb)).view(b, c, -1)
        phi_context_spot_own = self._apply_phi(context[:, 0, :]) 

        neighbor_contexts = context[:, 1:, :] 
        neighbor_masks = context_mask[:, 1:]  

        phi_neighbor_contexts_flat = self._apply_phi(neighbor_contexts.reshape(-1, d_ctx_emb))
        phi_neighbor_contexts = phi_neighbor_contexts_flat.view(b, n_total_ctx - 1, self.hparams.phi2rho_size)
        phi_neighbor_contexts_masked_for_sum = phi_neighbor_contexts.masked_fill(neighbor_masks.unsqueeze(-1), 0.0)
        sum_phi_neighbor_contexts = phi_neighbor_contexts_masked_for_sum.sum(dim=1)
        num_valid_neighbors = (~neighbor_masks).sum(dim=1, keepdim=True).float().clamp(min=1)
        mean_phi_neighbor_contexts = sum_phi_neighbor_contexts / num_valid_neighbors

        phi_context_spot_own_expanded = phi_context_spot_own.unsqueeze(1).expand(-1, c, -1)
        mean_phi_neighbor_contexts_expanded = mean_phi_neighbor_contexts.unsqueeze(1).expand(-1, c, -1)
        concatenated_per_cell = torch.cat([phi_cells, phi_context_spot_own_expanded, mean_phi_neighbor_contexts_expanded], dim=2)
        concatenated_per_cell = concatenated_per_cell * mask.unsqueeze(-1)
        aggregated_features = concatenated_per_cell.sum(dim=1)

        return self._apply_rho(aggregated_features)

    def _apply_phi(self, x):  
        return torch.median(torch.stack([p(x) for p in self.phi_spot], 1), 1).values
    
    def _apply_phi_cell(self, x):  
        return torch.median(torch.stack([p(x) for p in self.phi_cell], 1), 1).values
    
    def _apply_rho(self, x):  
        return torch.stack([r(x) for r in self.rho], 1).mean(1)

    def training_step(self, batch: Dict, batch_idx: int):
        cell_emb = batch['cell_embeddings'].float()
        spot_true = batch['spot_expression'].float()
        if self.cell_gt_available:
            cell_true = batch['cell_expressions'].float()
        mask = batch['cell_mask'].float()
        context = batch['neighb_embeddings'].float()
        context_mask = batch['neighb_masks'].bool()

        # pass only cells that are inside the spot: make outside cells zero
        is_inside = batch['is_inside_spot'].float()
        cell_emb_masked = cell_emb * is_inside.unsqueeze(-1)
        mask_inspot = mask * is_inside

        pred = self(cell_emb_masked, mask_inspot, context, context_mask)
        loss = self.loss_fn(pred, spot_true)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.train_batch_losses.append(loss.item())
        
        # Compute cell-level predictions for training metrics if ground truth available
        if self.cell_gt_available:
            for b in range(cell_emb.size(0)):
                valid = mask[b] > 0
                if valid.any():
                    spot_context = context[b]
                    spot_context_mask = context_mask[b]
                    
                    cell_preds = self._forward_single_cell(cell_emb[b, valid], spot_context, spot_context_mask).clamp_(min=0.0)
                    cell_targets = cell_true[b, valid]
                    is_inside_batch = is_inside[b, valid]
                    
                    # Separate inside and outside cells
                    inside_mask = is_inside_batch > 0
                    outside_mask = is_inside_batch == 0
                    
                    if inside_mask.any():
                        self.train_cell_inside_preds.append(cell_preds[inside_mask])
                        self.train_cell_inside_targets.append(cell_targets[inside_mask])
                    
                    if outside_mask.any():
                        self.train_cell_outside_preds.append(cell_preds[outside_mask])
                        self.train_cell_outside_targets.append(cell_targets[outside_mask])

        return loss


    def validation_step(self, batch: Dict, batch_idx: int):
        cell_emb = batch['cell_embeddings'].float()
        spot_true = batch['spot_expression'].float()
        if self.cell_gt_available:
            cell_true = batch['cell_expressions'].float()
        mask = batch['cell_mask'].float()
        context = batch['neighb_embeddings'].float()
        context_mask = batch['neighb_masks'].bool()
        is_inside = batch['is_inside_spot'].float()
        mask_inspot = mask * is_inside # mask out outside cells

        spot_pred = self(cell_emb * is_inside.unsqueeze(-1), mask_inspot, context, context_mask).clamp_(min=0.0)
        loss = self.loss_fn(spot_pred, spot_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.val_batch_losses.append(loss.item())

        if self.cell_gt_available:
            for b in range(cell_emb.size(0)):
                valid = mask[b] > 0
                if valid.any():
                    spot_context = context[b]
                    spot_context_mask = context_mask[b]

                    cell_preds = self._forward_single_cell(cell_emb[b, valid], spot_context, spot_context_mask).clamp_(min=0.0)
                    cell_targets = cell_true[b, valid]
                    is_inside_batch = is_inside[b, valid]
                    
                    # Separate inside and outside cells
                    inside_mask = is_inside_batch > 0
                    outside_mask = is_inside_batch == 0
                    
                    if inside_mask.any():
                        self.val_cell_inside_preds.append(cell_preds[inside_mask])
                        self.val_cell_inside_targets.append(cell_targets[inside_mask])

                    if outside_mask.any():
                        self.val_cell_outside_preds.append(cell_preds[outside_mask])
                        self.val_cell_outside_targets.append(cell_targets[outside_mask])

        return loss
    
    def on_train_epoch_end(self):
        if self.cell_gt_available:
            # Compute cell pearson correlations for inside cells
            if self.train_cell_inside_preds:
                all_preds = torch.cat(self.train_cell_inside_preds, dim=0)
                all_targets = torch.cat(self.train_cell_inside_targets, dim=0)
                pearson = safe_pearson(all_preds, all_targets)
                self.log('train_cell_inside_pearson', pearson.mean(), prog_bar=True)
            
            # Compute cell pearson correlations for outside cells
            if self.train_cell_outside_preds:
                all_preds = torch.cat(self.train_cell_outside_preds, dim=0)
                all_targets = torch.cat(self.train_cell_outside_targets, dim=0)
                pearson = safe_pearson(all_preds, all_targets)
                self.log('train_cell_outside_pearson', pearson.mean(), prog_bar=True)
        
        self.train_cell_inside_preds.clear()
        self.train_cell_inside_targets.clear()
        self.train_cell_outside_preds.clear()
        self.train_cell_outside_targets.clear()

    def on_validation_epoch_end(self):
        if self.cell_gt_available:
            # Compute cell pearson correlations for inside cells
            if self.val_cell_inside_preds:
                all_preds = torch.cat(self.val_cell_inside_preds, dim=0)
                all_targets = torch.cat(self.val_cell_inside_targets, dim=0)
                pearson = safe_pearson(all_preds, all_targets)
                self.log('val_cell_inside_pearson', pearson.mean(), prog_bar=True)
            
            # Compute cell pearson correlations for outside cells
            if self.val_cell_outside_preds:
                all_preds = torch.cat(self.val_cell_outside_preds, dim=0)
                all_targets = torch.cat(self.val_cell_outside_targets, dim=0)
                pearson = safe_pearson(all_preds, all_targets)
                self.log('val_cell_outside_pearson', pearson.mean(), prog_bar=True)
        
        self.val_cell_inside_preds.clear()
        self.val_cell_inside_targets.clear()
        self.val_cell_outside_preds.clear()
        self.val_cell_outside_targets.clear()
    
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {"optimizer": optimizer}