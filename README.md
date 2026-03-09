# DeepSpot: Leveraging Spatial Context for Enhanced Spatial Transcriptomics Prediction from H\&E Images

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![medRxiv](https://img.shields.io/badge/medRxiv-2025.02.09.25321567-blue)](https://www.medrxiv.org/content/10.1101/2025.02.09.25321567v3)
[![Data on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-orange)](https://huggingface.co/datasets/nonchev/TCGA_digital_spatial_transcriptomics)
[![Pretrained Models](https://img.shields.io/badge/Zenodo-Pretrained%20Weights-blue)](https://zenodo.org/records/15322099)

**Authors**: Kalin Nonchev, Sebastian Dawo, Karina Silina, Holger Moch, Sonali Andani, Tumor Profiler Consortium, Viktor Hendrik Koelzer, and Gunnar Rätsch

The preprint is available [here](https://www.medrxiv.org/content/10.1101/2025.02.09.25321567v3).

## News

  - [03.2026] Cross-sample [AESTETIK, the spatial transcriptomics integration model](https://github.com/ratschlab/aestetik) powering DeepSpot, will be presented at the [ICLR 2026 Learning Meaningful Representations of Life](https://www.biorxiv.org/content/10.64898/2026.03.02.709002v1).
  - [01.2026] Invited talk at 10x Genomics Single Cell & Spatial Discovery Symposium, 2026, Bern, Switzerland
  - [12.2025] Contributed talk at [NeurIPS 2025 Workshop on Multi-modal Foundation Models and Large Language Models for Life Sciences](https://nips2025fm4ls.github.io/), 2025, San Diego, USA
  - [10.2025] [DeepSpot2Cell: Predicting Virtual Single-Cell Spatial Transcriptomics from H&E images using Spot-Level Supervision](https://www.biorxiv.org/content/10.1101/2025.09.23.678121v1) at NeurIPS 2025 Imageomics. [Code.](https://github.com/ratschlab/DeepSpot2Cell)
  - [9.2025] DeepSpot was featured in the Eric and Wendy Schmidt Center article ["Machine Learning Teams Push the Boundaries of Virtual Spatial Biology in Global Autoimmune Disease Challenge."](https://www.ericandwendyschmidtcenter.org/updates/machine-learning-teams-push-the-boundaries-of-virtual-spatial-biology-in-global-autoimmune-disease-challenge)
  - [5.2025] Updated manuscript with new in-depth experiments and benchmarks, single-cell spatial transcriptomics prediction from H&E images, new datasets [link](https://www.medrxiv.org/content/10.1101/2025.02.09.25321567v2).
  - [04.2025] DeepSpot secured **1st place** at the Autoimmune ML Challenge organized by the Broad Institute of MIT and Harvard and CrunchDao for predicting single-cell spatial transcriptomics from H&E images [link](https://bmi.inf.ethz.ch/news/article/kalin-nonchev-wins-autoimmune-ml-challenge).

## Changelog

### NEW version (May 2025)
  - DeepCell for spatial transcriptomics at **single-cell** resolution
  - [new tutorials](example_notebook/)
  - change in the module structure
      - `from deepspot.spot import DeepSpot`
      - `from deepspot.cell import DeepCell`
  - [8 TB of predicted TCGA spatial transcriptomics data - LUSC, LUAD, SKCM, KIRC](https://huggingface.co/datasets/nonchev/TCGA_digital_spatial_transcriptomics)
    - 3780 samples
    - 56 317 393 spots
    - 4 cancer types
  - [more DeepSpot pretrained models on Visium and Xenium data](https://zenodo.org/records/15322099)



## Do you want to generate spatial transcriptomics data using your H&E images?

We introduce DeepSpot, a novel deep-learning model that predicts spatial transcriptomics from H&E images. DeepSpot employs a deep-set neural network to model spots as bags of sub-spots and integrates multi-level tissue details and spatial context. This integration, supported by the robust pre-trained H&E models, significantly enhances the accuracy and granularity of gene predictions from H&E images.

![deepspot](/figures/figure_2.png)

**Fig.: DeepSpot leverages pathology foundation models and spatial tissue context.**
**Workflow of DeepSpot**: H&E slides are first divided into tiles, each corresponding to a spot. For each spot, we create a bag of sub-spots by dividing it into sub-tiles that capture the local morphology, and a bag of neighboring spots to represent the global context. A pretrained pathology model extracts tile features, which are input to the model. The concatenated representations are then fed into the gene head predictor, ρgene, to predict spatial gene expression.

## Setup

```
git clone https://github.com/ratschlab/DeepSpot
cd DeepSpot
conda env create --file=environment.yaml
conda activate deepspot
python setup.py install
```

##### NB: Please ensure you have installed [pyvips](https://github.com/libvips/pyvips) depending on your machine's requirements. We suggest installing pyvips through conda:
```
conda install conda-forge::pyvips
```

Install jupyter kernel 
```
python -m ipykernel install --user --name deepspot --display-name "deepspot"
```

## Getting Started

Please take a look at our notebook collection to get started with DeepSpot for Visium or the adapted version of DeepSpot - DeepCell, for Xenium. We provide a small toy examples.

**DeepSpot** for spatial transcriptomics at **spot** resolution:
  - [Spatial transcriptomics data preprocessing spot level resolution](example_notebook/Visium_spot_example/GettingStartedWithDeepSpot_1_preprocessing.ipynb)
  - [DeepSpot training](example_notebook/Visium_spot_example/GettingStartedWithDeepSpot_2_training.ipynb)
  - [DeepSpot inference](example_notebook/Visium_spot_example/GettingStartedWithDeepSpot_3_inference.ipynb)
  - [DeepSpot inference with pretrained model](example_notebook/Visium_spot_example/GettingStartedWithDeepSpot_3.1_inference_pretrained_models.ipynb)

**DeepCell** for spatial transcriptomics at **single-cell** resolution:
  - [Spatial transcriptomics data preprocessing single-cell level resolution](example_notebook/Xenium_single-cell_example/GettingStartedWithDeepCell_1_preprocessing.ipynb)
  - [DeepCell training](example_notebook/Xenium_single-cell_example/GettingStartedWithDeepCell_2_training.ipynb)
  - [DeepCell inference](example_notebook/Xenium_single-cell_example/GettingStartedWithDeepCell_3_inference.ipynb)

## Pretrained DeepSpot weights

Moreover, we provide pretrained weights for DeepSpot, which were generated during the training of the model in our publication and were used, for example, to generate spatial transcriptomics data for TCGA skin melanoma and kidney cancer slides. 
Download DeepSpot weights [here](https://zenodo.org/records/15322099).

## Pathology foundation models

Please ensure that you download the weights for the pathology foundation models and update their file path deepspot/utils/utils_image.py. You may need to agree to specific terms and conditions before downloading.

   - UNI weights https://huggingface.co/MahmoodLab/UNI
   - Hoptimus0 weights https://huggingface.co/bioptimus/H-optimus-0
   - Phikon weights https://huggingface.co/owkin/phikon

## TCGA spatial transcriptomics data

We provide publicly the predicted spatial transcriptomics data with over 56 million spots from 3 780 TCGA patients with melanoma or kidney cancer. You can find the data [here](https://huggingface.co/datasets/nonchev/TCGA_digital_spatial_transcriptomics). Please navigate to the Hugging Face dataset card for more information.

### How to start?

```
pip install datasets
```

### Logging

```
from huggingface_hub import login, hf_hub_download, snapshot_download
import squidpy as sq
import pandas as pd
import scanpy as sc
import datasets

login(token="YOUR HUGGINGFACE TOKEN")
```

### Load metadata information

```
# Define dataset details
repo_id = "nonchev/TCGA_digital_spatial_transcriptomics"
filename = "metadata_2025-01-11.csv" # please check which is the latest one
```

```
# Create path
file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
# Load metata
metadata = pd.read_csv(file_path)
metadata.head()
```

```
        dataset slide_type                                          sample_id  n_spots                                          file_path
0     TCGA_SKCM       FFPE  TCGA-BF-AAP6-01Z-00-DX1.EFF1D6E1-CDBC-4401-A10...     5860  TCGA_SKCM/FFPE/TCGA-BF-AAP6-01Z-00-DX1.EFF1D6E...
1     TCGA_SKCM       FFPE  TCGA-FS-A1ZU-06Z-00-DX3.0C477EE6-C085-42BE-8BA...     2856  TCGA_SKCM/FFPE/TCGA-FS-A1ZU-06Z-00-DX3.0C477EE...
2     TCGA_SKCM       FFPE  TCGA-D9-A1X3-06Z-00-DX1.17AC16CC-5B22-46B3-B9C...     6236  TCGA_SKCM/FFPE/TCGA-D9-A1X3-06Z-00-DX1.17AC16C...
```

### Download a single TCGA spatial transcriptomics sample

```
local_dir = 'TCGA_data'  # Change the folder path as needed

snapshot_download("nonchev/TCGA_digital_spatial_transcriptomics", 
                  local_dir=local_dir,
                  allow_patterns="TCGA_SKCM/FFPE/TCGA-D9-A3Z3-06Z-00-DX1.C4820632-C64D-4661-94DD-9F27F75519C3.h5ad.gz",
                  repo_type="dataset")
```

```
adata = sc.read_h5ad("path/to/h5ad.gz")
sq.pl.spatial_scatter(adata, 
                      color=["SOX10", "CD37", "COL1A1", "predicted_label"],
                      size=20, img_alpha=0.8, ncols=2)
```
![example](/figures/tcga_example.png)

### Download the entire TCGA digital spatial transcriptomics dataset

```
local_dir = 'TCGA_data'  # Change the folder path as needed

# Note that the full dataset is around 7TB

snapshot_download("nonchev/TCGA_digital_spatial_transcriptomics", 
                  local_dir=local_dir,
                  repo_type="dataset")
```

#### NB: To distinguish in-tissue spots from the background, tiles with a mean RGB value above 200 (near white) were discarded. Additional preprocessing can remove potential image artifacts.

## Papers Citing DeepSpot

<!-- CITATIONS:START -->
1. Benjamin S Strope, D. Varghese, William Z Bowie, Stacy Wang, and Qian Zhu "CancerSTFormer enables multi-scale analysis of spot-resolution spatial transcriptomes and dissects gene and immune regulatory responses to targeted therapies." *bioRxiv* (2026). [DOI](https://doi.org/10.64898/2025.12.22.696102)
2. Justina Dai, Kalin Nonchev, V. Koelzer, and Gunnar Rätsch "Towards Cross-Sample Alignment for Multi-Modal Representation Learning in Spatial Transcriptomics." *bioRxiv* (2026). [DOI](https://doi.org/10.64898/2026.03.02.709002)
3. Ashley P. Tsang, S. Krishnan, Reva Kulkarni, Sagnik Bhadury, M. P. di Magliano, Timothy L. Frankel, and Arvind Rao "Encoding functional edges in graphs to model spatially varying relationships in the tumor microenvironment." *npj Artificial Intelligence* (2026). [DOI](https://doi.org/10.1038/s44387-026-00075-5)
4. Till Richter, Eric Zimmermann, J. Hall, Fabian J Theis, Srivatsan Raghavan, Peter S. Winter, Ava P. Amini, and Lorin Crawford "Beyond alignment: synergistic integration is required for multimodal cell foundation models." *bioRxiv* (2026). [DOI](https://doi.org/10.64898/2026.02.23.707420)
5. Kalin Nonchev, Glib Manaiev, V. Koelzer, and Gunnar Rätsch "DeepSpot2Cell: Predicting Virtual Single-Cell Spatial Transcriptomics from H&E images using Spot-Level Supervision." *bioRxiv* (2025). [DOI](https://doi.org/10.1101/2025.09.23.678121)
6. C. Delrue, and M. Speeckaert "Transcriptomic Signatures in IgA Nephropathy: From Renal Tissue to Precision Risk Stratification." *International Journal of Molecular Sciences* (2025). [DOI](https://doi.org/10.3390/ijms262010055)
7. Jonathan Xu, Michelle Jiang, Shunsuke Koga, Nancy R. Zhang, and Zhi Huang "SpatialFinder: A Human-in-the-Loop Vision-Language Framework for Prioritizing High-Value Regions in Spatial Transcriptomics." *bioRxiv* (2025). [DOI](https://doi.org/10.1101/2025.08.16.670684)
8. Moritz Schaefer, Kalin Nonchev, Animesh Awasthi, Jake Burton, V. Koelzer, Gunnar Rätsch, and Christoph Bock "Molecularly informed analysis of histopathology images using natural language." *bioRxiv* (2025). [DOI](https://doi.org/10.1101/2025.07.14.664402)
9. Sushant Patkar, Timothy R. Rosean, Palak Patel, Stephanie A. Harmon, Peter L. Choyke, Tamara Jamaspishvili, and B. Turkbey "Towards interpretable molecular and spatial analysis of the tumor microenvironment from digital histopathology images with HistoTME-v2." *bioRxiv* (2025). [DOI](https://doi.org/10.1101/2025.06.11.658673)
<!-- CITATIONS:END -->

*This list is automatically updated weekly via [GitHub Actions](.github/workflows/update-citations.yml) using the [Semantic Scholar](https://www.semanticscholar.org/) and [OpenCitations](https://opencitations.net/) APIs.*

## Related Projects

- [AESTETIK](https://github.com/ratschlab/aestetik) — AutoEncoder for learning multi-modal spatial transcriptomics representations. Powers cross-sample integration in DeepSpot.
- [DeepSpot2Cell](https://github.com/ratschlab/DeepSpot2Cell) — Predicts virtual single-cell spatial transcriptomics from H&E images using spot-level supervision. Presented at NeurIPS 2025 Imageomics.

## Citation

In case you found our work useful, please consider citing us:

```
@article{nonchev2025deepspot,
  title={DeepSpot: Leveraging Spatial Context for Enhanced Spatial Transcriptomics Prediction from H\&E Images},
  author={Nonchev, Kalin and Dawo, Sebastian and Silina, Karina and Moch, Holger and Andani, Sonali and Tumor Profiler Consortium and Koelzer, Viktor H and Raetsch, Gunnar},
  journal={medRxiv},
  pages={2025--02},
  year={2025},
  publisher={Cold Spring Harbor Laboratory Press}
}
```


The code for reproducing the paper results can be found [here](https://github.com/ratschlab/he2st).

## Contact

In case, you have questions, please get in touch with [Kalin Nonchev](https://bmi.inf.ethz.ch/people/person/kalin-nonchev).

#### NB: Computational data analysis was performed at Leonhard Med (https://sis.id.ethz.ch/services/sensitiveresearchdata/) secure trusted research environment at ETH Zurich. Our pipeline aligns with the specific cluster requirements and resources.
