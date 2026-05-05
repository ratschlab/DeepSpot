from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import json
import math

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_DIR.parent.parent
IMAGE_PATH = PROJECT_ROOT / "example_data" / "data" / "image" / "ZEN38.jpg"
META_PATH = PROJECT_ROOT / "example_data" / "data" / "meta" / "ZEN38.json"
H5AD_PATH = PROJECT_ROOT / "example_data" / "data" / "h5ad" / "ZEN38.h5ad"
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"

SPOT_INDEX = 222
NEIGHBOR_COUNT = 6

# Bounding boxes around single cells (x, y, width, height) in the low-res image space
CELL_BBOXES: List[Tuple[int, int, int, int]] = [
	(3524, 4294, 15, 11),
	(3525, 4304, 15, 7),
	(3534, 4306, 15, 11),
	(3511, 4325, 12, 12),
	(3516, 4335, 10, 12),
]


def _load_metadata() -> Dict:
	with META_PATH.open("r", encoding="utf-8") as fh:
		return json.load(fh)


def _compute_scaling(meta: Dict, image_size: Tuple[int, int]) -> Tuple[float, float]:
	fr_w = meta.get("fullres_px_width") or meta.get("fullres_width")
	fr_h = meta.get("fullres_px_height") or meta.get("fullres_height")
	if not fr_w or not fr_h:
		raise ValueError("Metadata missing full-resolution dimensions")
	img_w, img_h = image_size
	return img_w / float(fr_w), img_h / float(fr_h)


def _compute_spot_radius_fullres(meta: Dict) -> float:
	pix_um = meta.get("pixel_size_um_embedded") or meta.get("pixel_size_um_estimated") or meta.get("pixel_size_um")
	if meta.get("spot_diameter") and pix_um:
		return (float(meta["spot_diameter"]) / float(pix_um)) / 2.0
	if meta.get("spot_diameter_fullres"):
		return float(meta["spot_diameter_fullres"])
	raise ValueError("Metadata missing spot diameter information")


def _compute_interspot_fullres(meta: Dict) -> float:
	pix_um = meta.get("pixel_size_um_embedded") or meta.get("pixel_size_um_estimated") or meta.get("pixel_size_um")
	inter = meta.get("inter_spot_dist")
	if inter is None:
		raise ValueError("Metadata missing inter_spot_dist")
	inter_px = float(inter)
	if pix_um:
		inter_px = float(inter) / float(pix_um)
	return inter_px


def _get_full_coords(adata: ad.AnnData) -> np.ndarray:
	if "pxl_row_in_fullres" in adata.obs.columns and "pxl_col_in_fullres" in adata.obs.columns:
		rows = adata.obs["pxl_row_in_fullres"].astype(float).to_numpy()
		cols = adata.obs["pxl_col_in_fullres"].astype(float).to_numpy()
	elif "y_pixel" in adata.obs.columns and "x_pixel" in adata.obs.columns:
		rows = adata.obs["y_pixel"].astype(float).to_numpy()
		cols = adata.obs["x_pixel"].astype(float).to_numpy()
	else:
		raise ValueError("AnnData object missing pixel coordinate columns")
	return np.stack([rows, cols], axis=1)


def _find_neighbor_indices(full_coords: np.ndarray, spot_idx: int, count: int) -> List[int]:
	spot_coord = full_coords[spot_idx]
	dists = np.linalg.norm(full_coords - spot_coord[None, :], axis=1)
	order = np.argsort(dists)
	neighbors = [int(i) for i in order if i != spot_idx][:count]
	return neighbors


def _crop_square(image: Image.Image, center_xy: Tuple[float, float], side: float) -> Image.Image:
	half = side / 2.0
	left = int(round(center_xy[0] - half))
	top = int(round(center_xy[1] - half))
	right = int(round(center_xy[0] + half))
	bottom = int(round(center_xy[1] + half))
	left = max(0, left)
	top = max(0, top)
	right = min(image.width, right)
	bottom = min(image.height, bottom)
	return image.crop((left, top, right, bottom))


def _square_from_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
	x, y, w, h = bbox
	cx = x + w / 2.0
	cy = y + h / 2.0
	side = max(w, h)
	half = side / 2.0
	left = int(math.floor(cx - half))
	top = int(math.floor(cy - half))
	right = int(math.ceil(cx + half))
	bottom = int(math.ceil(cy + half))
	return left, top, right, bottom


def _save_image(image: Image.Image, path: Path) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	image.save(path)
	return path


def _display_context(image: Image.Image,
					 main_xy: Tuple[float, float],
					 neighbor_points: List[Tuple[float, float]],
					 cell_boxes: List[Tuple[int, int, int, int]],
					 spot_radius_img: float,
					 tile_side: float) -> None:
	annotated = image.copy()
	draw = ImageDraw.Draw(annotated)

	def _tile_rect(center: Tuple[float, float]) -> Tuple[float, float, float, float]:
		half = tile_side / 2.0
		left = max(0.0, center[0] - half)
		top = max(0.0, center[1] - half)
		right = min(float(annotated.width), center[0] + half)
		bottom = min(float(annotated.height), center[1] + half)
		return left, top, right, bottom

	# draw main spot (green) and neighbors (blue)
	main_tile_rect = _tile_rect(main_xy)
	draw.rectangle(main_tile_rect, outline=(0, 255, 0), width=4)
	draw.ellipse([
		main_xy[0] - spot_radius_img,
		main_xy[1] - spot_radius_img,
		main_xy[0] + spot_radius_img,
		main_xy[1] + spot_radius_img,
	], outline=(0, 255, 0), width=4)

	neighbor_tile_rects: List[Tuple[float, float, float, float]] = []
	for nx, ny in neighbor_points:
		tile_rect = _tile_rect((nx, ny))
		draw.rectangle(tile_rect, outline=(0, 120, 255), width=3)
		neighbor_tile_rects.append(tile_rect)
		draw.ellipse([
			nx - spot_radius_img,
			ny - spot_radius_img,
			nx + spot_radius_img,
			ny + spot_radius_img,
		], outline=(0, 120, 255), width=3)

	# draw cell bounding boxes (red)
	for box in cell_boxes:
		draw.rectangle(box, outline=(255, 0, 0), width=2)

	# determine display crop boundaries
	xs = [main_xy[0]] + [p[0] for p in neighbor_points]
	ys = [main_xy[1]] + [p[1] for p in neighbor_points]
	xs.extend([main_tile_rect[0], main_tile_rect[2]])
	ys.extend([main_tile_rect[1], main_tile_rect[3]])
	for left, top, right, bottom in neighbor_tile_rects:
		xs.extend([left, right])
		ys.extend([top, bottom])
	for left, top, right, bottom in cell_boxes:
		xs.extend([left, right])
		ys.extend([top, bottom])

	margin = tile_side
	left = max(0, int(min(xs) - margin))
	right = min(annotated.width, int(max(xs) + margin))
	top = max(0, int(min(ys) - margin))
	bottom = min(annotated.height, int(max(ys) + margin))

	cropped_display = annotated.crop((left, top, right, bottom))

	plt.figure(figsize=(6, 6))
	plt.imshow(cropped_display)
	plt.axis("off")
	plt.title("Spot 222: cells(red), main spot tile(green), neighbors(blue)")
	plt.show()


def get_example_spot_context() -> Tuple[List[Path], Path, List[Path]]:
	"""Prepare image tiles for the hard-coded spot and visualize the context."""

	if not IMAGE_PATH.exists():
		raise FileNotFoundError(f"Expected image file {IMAGE_PATH} not found")
	if not META_PATH.exists():
		raise FileNotFoundError(f"Expected metadata file {META_PATH} not found")
	if not H5AD_PATH.exists():
		raise FileNotFoundError(f"Expected AnnData file {H5AD_PATH} not found")

	meta = _load_metadata()
	adata = ad.read_h5ad(H5AD_PATH)
	image = Image.open(IMAGE_PATH).convert("RGB")

	scaling = _compute_scaling(meta, image.size)
	scale_x, scale_y = scaling
	spot_radius_full = _compute_spot_radius_fullres(meta)
	spot_radius_img = spot_radius_full * ((scale_x + scale_y) / 2.0)
	interspot_full = _compute_interspot_fullres(meta)
	tile_side_img = interspot_full * ((scale_x + scale_y) / 2.0)

	full_coords = _get_full_coords(adata)
	mapped_coords = np.column_stack([
		full_coords[:, 1] * scale_x,
		full_coords[:, 0] * scale_y,
	])

	main_xy = tuple(mapped_coords[SPOT_INDEX])
	neighbor_indices = _find_neighbor_indices(full_coords, SPOT_INDEX, NEIGHBOR_COUNT)
	neighbor_points = [tuple(mapped_coords[idx]) for idx in neighbor_indices]

	spot_img = _crop_square(image, main_xy, tile_side_img)
	spot_img_path = _save_image(spot_img, OUTPUT_DIR / f"spot_{SPOT_INDEX}.png")

	neighbor_paths: List[Path] = []
	for order, idx in enumerate(neighbor_indices, start=1):
		neighbor_img = _crop_square(image, mapped_coords[idx], tile_side_img)
		neighbor_path = _save_image(neighbor_img, OUTPUT_DIR / f"neighbor_{order}_idx_{idx}.png")
		neighbor_paths.append(neighbor_path)

	cell_paths: List[Path] = []
	clamped_boxes: List[Tuple[int, int, int, int]] = []
	for i, bbox in enumerate(CELL_BBOXES, start=1):
		square_box = _square_from_bbox(bbox)
		left, top, right, bottom = square_box
		left = max(0, left)
		top = max(0, top)
		right = min(image.width, right)
		bottom = min(image.height, bottom)
		square_box = (left, top, right, bottom)
		crop = image.crop(square_box)
		cell_path = _save_image(crop, OUTPUT_DIR / f"cell_{i}.png")
		cell_paths.append(cell_path)
		clamped_boxes.append(square_box)

	_display_context(image, main_xy, neighbor_points, clamped_boxes, spot_radius_img, tile_side_img)

	return cell_paths, spot_img_path, neighbor_paths
def _extract_embedding(image_path: Path, morphology_model, preprocess, device: str) -> torch.Tensor:
	image = Image.open(image_path)
	size = max(image.width, image.height)
	if image.width != image.height:
		image = image.resize((size, size))
	tensor = preprocess(image).to(device).unsqueeze(0)
	with torch.no_grad():
		embedding = morphology_model(tensor)
	return embedding.squeeze(0)


def predict_spot_expression_from_image_paths(cell_image_paths: List[Path],
											 spot_image_path: Path,
											 neighbor_image_paths: List[Path],
											 deepspot2cell_model,
											 morphology_model,
											 preprocess,
											 device: str,
											 gene_names: List[str] | None = None) -> Dict[str, float]:

	cell_embeddings = []
	for cell_path in cell_image_paths:
		emb = _extract_embedding(cell_path, morphology_model, preprocess, device)
		cell_embeddings.append(emb)

	cell_tensor = torch.stack(cell_embeddings).unsqueeze(0).to(device)
	cell_mask = torch.ones((1, cell_tensor.size(1)), device=device)

	spot_embedding = _extract_embedding(spot_image_path, morphology_model, preprocess, device)
	context_embeddings = [spot_embedding]
	context_mask_flags = [False]

	for neighbor_path in neighbor_image_paths:
		if Path(neighbor_path).exists():
			context_embeddings.append(_extract_embedding(neighbor_path, morphology_model, preprocess, device))
			context_mask_flags.append(False)
		else:
			zeros = torch.zeros_like(spot_embedding)
			context_embeddings.append(zeros)
			context_mask_flags.append(True)

	context_tensor = torch.stack(context_embeddings).unsqueeze(0).to(device)
	context_mask = torch.tensor(context_mask_flags, dtype=torch.bool, device=device).unsqueeze(0)

	with torch.no_grad():
		predictions = deepspot2cell_model._forward_superspot(
			cell_tensor,
			cell_mask,
			context_tensor,
			context_mask
		).cpu().numpy()

	predictions = predictions.squeeze(0)
	results: Dict[str, float] = {}
	if gene_names and len(gene_names) == predictions.shape[-1]:
		for gene, value in zip(gene_names, predictions):
			results[gene] = float(value)
	else:
		for idx, value in enumerate(predictions):
			results[f"gene_{idx}"] = float(value)

	return results

