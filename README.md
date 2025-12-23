# Multi-Modal Weighted Fusion for Low-Light Image Enhancement

This repository contains the official implementation for the course project  
**“Multi-Modal Weighted Fusion for Low-Light Image Enhancement”**,  
developed at **National Yang Ming Chiao Tung University (NYCU)**.

The proposed method enhances low-light images by adaptively fusing a
detail-preserving branch and a noise-suppressing branch using
structure-aware and gradient-aware modality masks.

---

## Repository Structure

```
root: /workspace/MultiModel
├── dataset/            # Input low-light images
├── eval15/             # Evaluation dataset (LOL benchmark subset)
├── fusion_core.py      # Pixel-wise weighted fusion implementation
├── modalities.py       # Construction of M2 (gradient) and M3 (structure) masks
├── utils.py            # Utility and helper functions
├── main.ipynb          # Main execution notebook (entry point)
├── log.txt             # Optional execution logs
```

---

## Environment Requirements

- Python 3.10+
- Core dependencies:
  - numpy
  - opencv-python
- Notebook execution:
  - jupyter
- Evaluation and visualization:
  - scikit-image
  - matplotlib
- Perceptual metric (optional):
  - torch
  - lpips

Install dependencies with:

```bash
pip install numpy opencv-python scikit-image matplotlib jupyter torch lpips
```

---

## How to Run

### Step 1: Clone the Repository

```bash
git clone https://github.com/joshua12390902/Multi-Modality_Final.git
cd Multi-Modality_Final
```

### Step 2 (Optional): Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Step 3: Run the Main Pipeline

All experiments, visualizations, and evaluations are executed sequentially
inside the main notebook.

```bash
jupyter notebook main.ipynb
```

Running `main.ipynb` will:

- Construct gradient (M2) and structure (M3) modality masks
- Perform weighted fusion of detail and smooth branches
- Generate qualitative results
- Compute PSNR, SSIM, and LPIPS metrics

No additional configuration is required.

---

## Method Overview

### Detail Branch
Enhances local contrast using CLAHE to preserve fine structures.

### Smooth Branch
Applies denoising (NLM or BM3D) to suppress noise in flat regions.

### Gradient Mask (M2)
Computed using Canny edge detection followed by Gaussian diffusion
to softly emphasize structural boundaries.

### Structure Mask (M3)
Estimated from local intensity standard deviation with noise-floor subtraction
to distinguish texture from noise-dominated areas.

### Fusion Strategy
A pixel-wise weight map adaptively blends the two branches to achieve
a balanced trade-off between noise suppression and detail preservation.

---

## Reproducibility

All results reported in the accompanying paper can be reproduced
by directly running `main.ipynb`.

The code is designed for clarity and educational purposes,
with explicit implementation of each modality and fusion step.

---

## Course Information

- Institution: National Yang Ming Chiao Tung University
- Program: Institute of Intelligent and Computing Technology
- Course: Multi-Modality (Final Project)
