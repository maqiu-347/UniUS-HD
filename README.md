# MiT-B5 Multi-task Baseline with Masked Reconstruction Pretraining

This repository implements a shared MiT-B5 encoder for ultrasound segmentation, classification, landmark regression, and single-object localization. Segmentation and detection use a shared SMP FPN decoder. Detection uses a custom Faster R-CNN-style head with an RPN and RoI Align on one fused FPN feature map.

The original repository description associates this baseline with the manuscript *UniUS-HD: Unified Ultrasound Analysis with Hierarchical Representation Learning and Dynamic Boundary Refinement*. The supplied source files do not establish a particular table row or reproduce its reported results. This implementation does not contain the manuscript's MPM, PAL, or DBR modules. It should not be presented as the complete UniUS-HD framework.

**Implementation status:** the documentation reflects a static review of the supplied code. Full training and evaluation have not been validated. In particular, the current optimizer omits the FPN decoder, and the pretraining implementation has initialization and masking differences that must be resolved before claiming faithful experimental reproduction. See [review notes](CODE_REVIEW_NOTES.md).


## Architecture

### Overall UniUS-HD framework

![Overall UniUS-HD architecture showing masked pretraining and downstream multitask fine-tuning](docs/image/architecture.png)

The overall framework has two stages. Pretraining combines multi-scale progressive masking with multi-level reconstruction supervision and masked-view representation alignment. Downstream fine-tuning uses the shared MiT-B5 encoder for classification, landmark regression, segmentation, and detection, with dynamic boundary refinement in the detection training branch.

**Scope of this repository:** the figure presents the complete proposed UniUS-HD framework. The code and released checkpoints documented here belong to the ablation baseline, which does not implement MPM, PAL, or DBR. The encoder-freezing symbol in the overall figure should not be interpreted as the baseline training configuration: the reviewed `train.py` includes encoder parameters in its optimizer. 


## Model and task configuration

Training and inference use `mit_b5` with three input channels and images resized to 512 by 512. Classification and regression operate on the deepest encoder features. Only segmentation and detection pass through the shared FPN decoder.

The task registry contains 27 task IDs:

- Segmentation, 12 tasks: `breast_lesion`, `cardiac_multi`, `carotid_artery`, `cervix`, `cervix_multi`, `fetal_abdomen_multi`, `fetal_head`, `fetal_heart`, `head_symphysis_multi`, `lung`, `ovary_tumor`, `thyroid_nodule`.
- Classification, 9 tasks: `breast_2cls`, `breast_3cls`, `fetal_head_pos_cls`, `fetal_plane_cls`, `fetal_sacral_pos_cls`, `liver_lesion_2cls`, `lung_2cls`, `lung_disease_3cls`, `organ_cls`.
- Landmark regression, 3 tasks: `FUGC`, `IUGC`, `fetal_femur`. Their point counts are 2, 3, and 2 respectively.
- Detection, 3 tasks: `spinal_cord_injury_loc`, `thyroid_nodule_det`, `uterine_fibroid_det`.

CSV task names are case-sensitive: `segmentation`, `classification`, `Regression`, and `detection`. The global registry also controls forward routing, so arbitrary new IDs cannot be added through the inference CSV alone.

The detection pipeline currently uses one foreground class and one ground-truth box per sample. It returns one highest-scoring box per image, or a zero box with zero confidence when no proposal remains. It is not a general multi-object detection implementation. Anchor sizes are 16, 32, 64, 128, and 256 pixels, with aspect ratios 0.5, 0.75, 1.0, 1.5, and 2.0. The head assumes a 512 by 512 image size unless explicitly reconfigured. Its `box_score_thresh` setting is stored but not applied during inference.

`FPNGridDetectionHead` and the grid-based detection losses remain in the source as legacy components. The active factory creates `FPNTwoStageDetectionHead`; `train.py` uses the loss dictionary returned by that head.


## Released checkpoints

The baseline checkpoints are available as assets of [release v1.0.0](https://github.com/maqiu-347/UniUS-HD/releases/tag/v1.0.0):

- [mae_tradition_mit.pth](https://github.com/maqiu-347/UniUS-HD/releases/download/v1.0.0/mae_tradition_mit.pth): the masked reconstruction pretraining checkpoint, used to initialize the encoder for downstream training through `MAE_PRETRAINED_PATH`.
- [best_model_base.pth](https://github.com/maqiu-347/UniUS-HD/releases/download/v1.0.0/best_model_base.pth): the downstream baseline checkpoint, used for inference through `MODEL_PATH`.

Download them into `weights/` from the repository root. The following PowerShell commands preserve the release filenames:

```powershell
New-Item -ItemType Directory -Force -Path './weights' | Out-Null
Invoke-WebRequest -Uri 'https://github.com/maqiu-347/UniUS-HD/releases/download/v1.0.0/mae_tradition_mit.pth' -OutFile './weights/mae_tradition_mit.pth'
Invoke-WebRequest -Uri 'https://github.com/maqiu-347/UniUS-HD/releases/download/v1.0.0/best_model_base.pth' -OutFile './weights/best_model_base.pth'
```

To use the pretraining checkpoint for downstream training:

```powershell
$env:DATA_ROOT = './data/train'
$env:MAE_PRETRAINED_PATH = './weights/mae_tradition_mit.pth'
$env:MODEL_SAVE_PATH = './weights/best_model_base.pth'
python train.py
```

Training writes a new checkpoint to `MODEL_SAVE_PATH`; the example keeps the downloaded `best_model_base.pth` separate. If the MAE file does not exist, the reviewed training script falls back to ImageNet initialization, so check that the checkpoint was loaded successfully.

To run inference with the released baseline checkpoint:

```powershell
$env:DATA_ROOT = './data/val'
$env:MODEL_PATH = './weights/best_model_base.pth'
$env:OUTPUT_DIR = './predictions'
$env:BATCH_SIZE = '8'
python model.py
```

Set `MODEL_PATH` explicitly because the reviewed inference script defaults to `./best_model.pth`, which differs from the release filename. It expects a full model state dictionary, whereas the training loader accepts an encoder state dictionary or a checkpoint containing an `encoder` entry. Inference constructs task heads from its CSVs and loads weights strictly, so task IDs and class counts must match the checkpoint configuration. For a checkpoint containing all 27 task heads, a CSV defining only a subset can cause loading errors.

These links were supplied by the repository author. The checkpoint contents, integrity, loading compatibility, and evaluation results have not been verified in this documentation update. Their availability does not resolve the implementation limitations described above and in the review notes.

GitHub Release assets are distributed separately from files tracked in Git. The `.gitignore` rules intentionally exclude downloaded `.pth` files and the `weights/` directory; no exception is needed for either release checkpoint.

## Environment

Install a compatible PyTorch and torchvision pair for your hardware, followed by the repository dependencies:

```text
python -m pip install -r requirements.txt
python -m pip check
```

The current dependency file alone does not guarantee API compatibility. `model_factory.py` calls `fpn_decoder(features)` with a feature list; `mae.py` uses `RandomResizedCrop(size=...)`; `train.py` uses `BboxParams(clip=True)`. Select versions supporting these exact interfaces and record the resolved environment after validation. Do not assume that the oldest versions allowed by the requirements file support the code.

The scripts request ImageNet weights during model construction, including inference before loading the final checkpoint. An offline run needs those weights cached or a code change to initialization. CPU execution is selected automatically when CUDA is unavailable, but no CPU runtime or memory requirement has been benchmarked.

## Data format

Each data root must contain a `csv_files` directory. All `*.csv` files immediately inside that directory are concatenated. Image and mask paths are resolved relative to `csv_files`, not relative to the repository root.

```text
data/train/
    csv_files/
        tasks.csv
    images/
        example.png
    masks/
        example.png
```

For this layout, an image path in the CSV is `../images/example.png`. Every row should provide `image_path`, `task_id`, `task_name`, and an integer `num_classes` consistent with the task registry. Additional annotations are:

- Segmentation: `mask_path`, with a single-channel class-index mask. Values must be integers from 0 to `num_classes - 1`, where 0 is background. A 0/255 binary mask needs conversion to 0/1 before use.
- Classification: `mask`, containing the zero-based integer class label. This column is not an image path.
- Regression: `point_1_xy`, `point_2_xy`, and additional point columns where required. Each is a JSON list such as `[120, 85]` in original-image pixels. In a CSV cell, quote the list as `"[120, 85]"`. Here `num_classes` means the number of landmarks.
- Detection: `x_min`, `y_min`, `x_max`, `y_max` in original-image pixels. One box is read per row.

Images are loaded through OpenCV and converted to RGB. Supervised training applies resize, brightness/contrast augmentation, Gaussian noise, ImageNet normalization, and tensor conversion. Validation and inference omit the stochastic image augmentations. Regression targets are normalized using the original image dimensions; detection targets are normalized after the resize transform.

Provide complete, readable inputs. The current loaders recursively substitute another sample when an image cannot be read. Missing segmentation labels may become background masks, and missing landmark labels become zero coordinates. Invalid detection boxes may become `[-1, -1, -1, -1]`, which the active two-stage head does not explicitly exclude. These behaviors are not annotation validation or support for unlabeled samples.


## Reproducibility and release notes

Two baseline checkpoints are linked in the Released checkpoints section and distributed through GitHub Releases rather than tracked in the source repository. The supplied source files do not include a dataset, experiment log, or independently verified result table. Record the exact data split, patient grouping, pretraining pool, dependency versions, random seeds, checkpoint, and metric definitions for each reported run. If pretraining and supervised splitting use the same data root, check whether downstream validation images were included in pretraining and describe the resulting protocol explicitly.

The baseline release is identified as v1.0.0. A specific manuscript table row and the reported performance still require corresponding experiment records; the release links alone do not establish these claims. No release schedule for the complete UniUS-HD framework is asserted here. No license grant or dataset redistribution permission is established by these files.

`.gitignore` excludes common local artifacts by name. It does not remove files that Git already tracks, inspect file contents, or cover arbitrary paths selected through environment variables. Review staged files before publishing, and keep private data and checkpoints in the designated ignored directories. The supplied directory was not a Git repository at review time, so no tracked-file history was available to audit.
