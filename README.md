# MiT-B5 + Faster R-CNN + Traditional MAE Baseline

## Disclaimer
This repository provides the **ablation baseline** used in the paper
"UniUS-HD: Unified Ultrasound Analysis with Hierarchical Representation Learning and Dynamic Boundary Refinement".

It corresponds to the **`xxx` row in Table 2** of the paper:
- Encoder: MiT-B5 (ImageNet init -> traditional MAE pretraining -> fine-tuning)
- Decoder: SMP FPN
- Detection head: two-stage Faster R-CNN (RPN + RoI Heads)
- Pretraining: **traditional MAE** with fixed patch size 32, fixed mask ratio 0.75, pixel reconstruction loss only
- **No** multi-scale progressive masking (MPM)
- **No** pyramid reconstruction / masked-view alignment (PAL)
- **No** dynamic boundary refinement (DBR)

This is **NOT** the full UniUS-HD framework, and **NOT** the `FMC_UIA Baseline`
in Table 1 (that one uses EfficientNet-B4 + grid detection).
Full UniUS-HD code and pretrained weights will be released upon acceptance.

## Repository layout