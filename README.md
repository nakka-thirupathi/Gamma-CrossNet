# GAMMA-CrossNet: PyTorch research implementation

This project implements the manuscript's high-level pipeline:
Swin Transformer encoder -> three-parameter Gamma-guided representation ->
cross-attention (Q from Swin; K,V from Gamma features) -> U-Net decoder ->
predicted lesion ROI -> PVT classifier.

## Important mathematical reproducibility note

The uploaded Supplementary Section S1 provides the three-parameter Gamma PDF:

`p(x|alpha,beta,mu) = (x-mu)^(alpha-1) exp(-(x-mu)/beta) / (Gamma(alpha) beta^alpha), x > mu`

It defines `T`, `E`, and `G` as functions but does NOT specify their actual operations:
- S2 does not state the exact positive feature transform T.
- S3 does not choose MLE, moments, or another estimator E.
- S4 does not give the actual Gamma-guided feature-generation function G.


## Environment

Python 3.10/3.11 recommended. Install:
```
pip install -r requirements.txt
```

## Dataset manifest

Create one CSV per split with columns:
- `image`: path to mammogram image
- `mask`: path to binary lesion mask (blank only for classification-only data)
- `label`: integer class (0=benign, 1=malignant; confirm this mapping)
- `case_id`: patient/examination identifier used for grouped splitting

Important: CBIS-DDSM and MIAS do not automatically provide identical mask/label
structures. Prepare dataset-specific annotation mappings and verify all mask
files manually. MIAS has 322 digitized mammograms, but lesion masks must be
constructed only from documented annotations or a verified annotation source.

## Create splits

```
python scripts/make_case_splits.py --csv metadata/all_cases.csv --out splits
```
Inspect case-level disjointness, label counts and mask availability. For final
paper reproduction, use the exact split identifiers from the original study,
if available; this utility creates a new split and cannot recover the original.

## Train and evaluate

```
python train.py --train_csv splits/train.csv --val_csv splits/val.csv
python evaluate.py --test_csv splits/test.csv --checkpoint checkpoints/best.pt
```

Use a small batch size if GPU memory is limited. The default config uses batch
size 2 for broad hardware compatibility; the manuscript's reported batch size
should be entered only after confirmation from the actual experiment log.

## Ablations

The model forward pass supports:
- `use_gamma=False`
- `use_cross_attention=False`

For a formal ablation, train a separate model from scratch for each variant,
using identical splits, seeds, preprocessing, epochs, and optimization settings.
Do not merely disable a module at test time and call that a trained ablation.
The ViT encoder and CNN classifier replacement variants require corresponding
backbone substitutions and controlled retraining.

## Metrics and limitations

Classification metrics include accuracy, precision, sensitivity, specificity,
F1 and ROC-AUC. Segmentation metrics currently include Dice and IoU. HD95 and
ASD require robust surface-distance implementation and should be added before
reporting those metrics. The code does not hardcode any manuscript result.

The preprocessing function includes a documented median-filter fallback, not
a Wiener filter implementation. Replace it with the exact Wiener configuration
used in the experiment if that was indeed used. Record library versions, seeds,
GPU, checkpoint hashes and per-case predictions for a reproducible audit trail.
