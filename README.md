# MiT-B5 with a Two-Stage Detection Head and MAE-Style Pretraining

Documentation for the supplied `mit_faster_mae_tradition` implementation associated with **UniUS-HD: Unified Ultrasound Analysis with Hierarchical Representation Learning and Dynamic Boundary Refinement**.

This repository contains an **ablation baseline**, not the complete UniUS-HD implementation. It combines a shared MiT-B5 encoder, a shared SMP FPN decoder for dense tasks, task-specific prediction heads, and fixed-scale masked reconstruction pretraining. The current files do not implement PAL, MPM, or DBR. The implementation is also distinct from the FMC-UIA official baseline.

No paper-table row number, numerical reproduction result, release date, or checkpoint availability is asserted here. These must be linked to an identified experiment and source version before publication.
# MiT-B5 Multi-task Baseline with Masked Reconstruction Pretraining

This repository implements a shared MiT-B5 encoder for ultrasound segmentation, classification, landmark regression, and single-object localization. Segmentation and detection use a shared SMP FPN decoder. Detection uses a custom Faster R-CNN-style head with an RPN and RoI Align on one fused FPN feature map.

The original repository description associates this baseline with the manuscript *UniUS-HD: Unified Ultrasound Analysis with Hierarchical Representation Learning and Dynamic Boundary Refinement*. The supplied source files do not establish a particular table row or reproduce its reported results. This implementation does not contain the manuscript's MPM, PAL, or DBR modules. It should not be presented as the complete UniUS-HD framework.

**Implementation status:** the documentation reflects a static review of the supplied code. Full training and evaluation have not been validated. In particular, the current optimizer omits the FPN decoder, and the pretraining implementation has initialization and masking differences that must be resolved before claiming faithful experimental reproduction. See [review notes](CODE_REVIEW_NOTES.md).

## Repository layout

- `mae.py`: masked image reconstruction pretraining and checkpoint saving.
- `dataset.py`: CSV-backed supervised dataset and task-uniform batch sampler.
- `model_factory.py`: task definitions, shared encoder, FPN decoder, and task heads.
- `faster_rcnn_head.py`: anchors, RPN, proposal selection, RoI head, and detection losses.
- `train.py`: supervised training and internal validation.
- `utils.py`: batching, validation metrics, seeds, and legacy detection utilities.
- `model.py`: checkpoint loading, inference, mask export, and JSON export.
- `evaluate.py`: evaluation of exported predictions against CSV annotations.
- `requirements.txt`: dependency declarations; these are broad lower bounds, not a validated environment lock.
- `.gitignore`: exclusions for local data, weights, environments, and generated files.

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

## Pretraining behavior

`mae.py` uses 512 by 512 inputs, patch size 32, masking ratio 0.75, and a reconstruction decoder with 8 blocks, dimension 512, and 16 attention heads. The objective is masked pixel MSE with per-patch target normalization enabled. The default run uses batch size 20 and 30 epochs with AdamW, 5 warmup epochs, and cosine scheduling. The base learning rate is scaled by batch size divided by 256.

This is a **MiT-based masked reconstruction implementation**, not an exact implementation of the original MAE algorithm. It first masks image patches before the encoder, then independently samples a second mask over encoded tokens. Reconstruction loss uses the second mask. The two masks are not synchronized.

The encoder is initially requested with ImageNet weights. However, `initialize_weights()` subsequently traverses the whole model and reinitializes its `Linear` and `LayerNorm` modules, including those in the encoder. Therefore, the supplied implementation does not preserve all loaded ImageNet parameters.

Downstream training loads the MAE encoder only when `MAE_PRETRAINED_PATH` exists. Otherwise it continues with the ImageNet-initialized downstream encoder. MAE pretraining is therefore optional in the current script, rather than an enforced stage.

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

## Running the scripts

The following commands describe the existing entry points. They do not resolve the implementation issues listed below. Run them from the repository root after configuring paths, dependencies, data, and checkpoints.

### Masked reconstruction pretraining

Edit `DATA_ROOT` and `SAVE_PATH` near the top of `mae.py`; they are hard-coded local paths and do not read environment variables. Create the parent directory of `SAVE_PATH` before running:

```text
python mae.py
```

On Windows, the local lambda used as `collate_fn` is incompatible with the configured multiprocessing worker setup. Replace it with a module-level function or set that loader's `num_workers` to 0 before use.

The checkpoint selected by lowest mean training reconstruction loss contains `encoder`, `model_state_dict`, `epoch`, and `loss`. It does not contain optimizer or scheduler states for exact training resumption. Pretraining does not create an independent validation split.

### Supervised training

PowerShell example:

```powershell
$env:DATA_ROOT = './data/train'
$env:MAE_PRETRAINED_PATH = './weights/mae_tradition_mit.pth'
$env:MODEL_SAVE_PATH = './weights/best_model_weight.pth'
python train.py
```

These values match the defaults. Other training settings are constants in `train.py`, including 50 epochs, batch size 20, seed 42, and a 20% validation split. The split is random at the CSV-row level, without patient grouping or task stratification. Training samples tasks uniformly at random per batch; this does not ensure equal task counts within every epoch. Very small task subsets may produce short batches.

The configured losses are multiclass Dice for segmentation, cross-entropy for classification, coordinate MSE for regression, and the sum of RPN objectness, RPN box, RoI classification, and RoI box losses for detection. AdamW uses an encoder learning rate of `1e-4`, task-head learning rates of `3e-4`, weight decay `0.01`, and cosine annealing. **The current parameter groups omit `model.fpn_decoder`, so its parameters are not updated by this optimizer.**

The checkpoint is a raw model `state_dict`. Its selection score averages selected validation metric columns, excludes the regression MAE column, and includes detection confidence. This is an implementation-specific selection rule, not a verified manuscript or challenge score. The training split is not exported, and the checkpoint does not support exact optimizer-state resumption.

### Inference

```powershell
$env:DATA_ROOT = './data/val'
$env:MODEL_PATH = './weights/best_model_weight.pth'
$env:OUTPUT_DIR = './predictions'
$env:BATCH_SIZE = '8'
python model.py
```

Set `MODEL_PATH` explicitly: inference defaults to `./best_model.pth`, whereas training defaults to `./weights/best_model_weight.pth`. Inference builds task heads from its input CSVs and loads the checkpoint strictly. With a checkpoint produced by the unmodified 27-task training script, the inference configuration must contain all matching heads and class counts. A task subset can otherwise cause unexpected checkpoint keys.

Segmentation outputs are class-index masks resized to the original image size. A supplied `mask_path` determines the output path after literal `../` removal. Always provide suitable relative segmentation mask paths. This string replacement is not a containment check; absolute paths, Windows backslash traversal, and the fallback based on `image_path` can write outside the requested output directory. Use a separate prediction directory and validate resolved output paths before execution.

For tasks present in the inputs, inference writes:

- `classification_predictions.json`: predicted classes and class probabilities.
- `detection_predictions.json`: one box in normalized and original-pixel coordinates, with confidence.
- `regression_predictions.json`: landmark coordinates in normalized and original-pixel units. The regression head has no sigmoid or clipping, so predictions can fall outside the image.

### Exported-prediction evaluation

Edit `data_root`, `pred_root`, `output_file`, and `summary_file` in the main block of `evaluate.py`. These are hard-coded paths, not environment-variable options. Create the result parent directories before running:

```text
python evaluate.py
```

This evaluator reports foreground-class DSC and Hausdorff distance for segmentation, AUC/accuracy/macro-F1/weighted-F1/MCC for classification, mean IoU for detection, and mean radial error for landmarks. HD and MRE are in pixel units; the script does not load physical spacing. HD is the maximum Hausdorff distance returned by SimpleITK, not HD95. Detection evaluation does not compute AP or mAP.

The evaluation summary averages task scores within each task type. Missing predictions are skipped rather than scored as failures, so inspect sample coverage. AUC failures can be represented as zero, and per-class F1 labels can be misassigned when class IDs are absent from the evaluated samples. These cases require correction before reporting final results.

Internal validation in `utils.py` is a separate calculation: its Dice aggregates foreground counts within each task batch, and its regression MAE uses a fixed 256 by 256 coordinate scale. This differs from both the 512 by 512 model input and the original-image radial error used in `evaluate.py`. Internal metrics are averaged over collected batches rather than recomputed from all predictions.

## Reproducibility and release notes

The supplied files contain no dataset, trained checkpoint, experiment log, or verified result table. Record the exact data split, patient grouping, pretraining pool, dependency versions, random seeds, checkpoint, and metric definitions for each reported run. If pretraining and supervised splitting use the same data root, check whether downstream validation images were included in pretraining and describe the resulting protocol explicitly.

The original README's unfilled table reference and promised release schedule have been removed because the supplied artifacts do not substantiate them. Add verified manuscript references and release details when available. No license grant or dataset redistribution permission is established by these files.

`.gitignore` excludes common local artifacts by name. It does not remove files that Git already tracks, inspect file contents, or cover arbitrary paths selected through environment variables. Review staged files before publishing, and keep private data and checkpoints in the designated ignored directories. The supplied directory was not a Git repository at review time, so no tracked-file history was available to audit.

**Current status:** the documentation was checked against the supplied source files. Training, inference, dependencies, and paper results have not been validated by this documentation work. In particular, the optimizer omits the shared FPN decoder, the fine-tuning split is sample-based rather than patient-based, and the pretraining implementation has initialization and masking details that require review. Read [Known issues](docs/KNOWN_ISSUES.md) before using this snapshot for a new experiment or a reproduction claim.

## What is implemented

- **Shared encoder:** MiT-B5 through `segmentation_models_pytorch`.
- **Segmentation:** a shared SMP FPN decoder followed by a separate segmentation head for each task.
- **Detection:** a custom two-stage head with RPN, RoI processing, and NMS. Each task head consumes one fused FPN feature tensor and exports one best box per image. This is not a complete torchvision Faster R-CNN model or a multi-level RoI FPN implementation.
- **Classification:** average pooling of the deepest encoder feature, dropout, and a task-specific classifier.
- **Regression:** average pooling of the deepest encoder feature followed by a linear coordinate head. Its outputs are not constrained with sigmoid or clipping.
- **Pretraining:** fixed patch size 32, mask ratio 0.75, and normalized pixel reconstruction loss. Pixels are masked before the dense encoder, and a second independently sampled mask is applied to encoder-output tokens. This should be described as the implementation's MAE-style baseline, not an exact reproduction of canonical MAE.

The factory defines 27 task IDs: 12 segmentation, 9 classification, 3 detection, and 3 regression. See the [task list and data specification](docs/DATA_FORMAT.md).

## Repository layout

```text
.
├── dataset.py           # CSV loading, label processing, uniform task sampling
├── mae.py               # Fixed-scale masked reconstruction pretraining
├── model_factory.py     # Shared encoder, FPN, 27 task configurations and heads
├── faster_rcnn_head.py  # Custom RPN and RoI detection implementation
├── train.py             # Joint downstream fine-tuning and internal validation
├── model.py             # Checkpoint loading, inference and prediction export
├── evaluate.py          # Evaluation of exported masks and JSON predictions
├── utils.py             # Internal validation metrics and helper functions
└── requirements.txt     # Existing dependency constraints, not a tested lockfile

```

## Installation

Run the following from the repository root in an isolated Python environment:

```bash
python -m venv .venv
# Linux or macOS
source .venv/bin/activate
# Windows PowerShell instead
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use a compatible PyTorch and torchvision installation for your hardware. The existing requirements specify broad lower bounds; they do not establish that every permitted combination works. In particular, verify the SMP FPN decoder API, Albumentations arguments, and torchvision NMS/RoI operations before a long run. GPU memory requirements and a tested GPU/software combination have not been established for this supplied snapshot.

Encoder construction requests ImageNet weights, including during inference before the full model checkpoint is loaded. Prepare the required encoder cache for offline operation. The pretraining initializer subsequently modifies some encoder parameters, as described in the known issues.

## Prepare data

Each dataset root must contain `csv_files/*.csv`. Paths in the CSV are resolved relative to that **csv_files directory**, not relative to the repository or dataset root.

```text
data/train/
├── csv_files/
│   └── tasks.csv
├── images/
└── masks/
```

For an image at `data/train/images/example.png`, use `../images/example.png` in the CSV. Use class-index masks with background 0, zero-based classification labels, original-image pixel coordinates for landmarks and boxes, and the exact task name `Regression` for landmark tasks.

The CSVs under `examples/` describe the schema only. They do not reference real images and are not a runnable benchmark. Do not place them in a real training directory. Data and checkpoints are not included in this documentation package.

## Run pretraining

`mae.py` does not read configuration from environment variables or command-line flags. Edit the `DATA_ROOT` and `SAVE_PATH` constants near the top of that file to point to an authorized image pool and an encoder checkpoint path, for example:

```python
DATA_ROOT = './data/pretrain'
SAVE_PATH = './weights/mae_tradition_mit.pth'
```

Create the output directory before running:

```bash
python -c "from pathlib import Path; Path('weights').mkdir(parents=True, exist_ok=True)"
python mae.py
```

The checkpoint contains `encoder`, `model_state_dict`, `epoch`, and `loss`. It is selected by training reconstruction loss, not held-out validation loss. On Windows, the pretraining loader's lambda collate function with four workers needs adjustment before use; see the known issues. For an inductive evaluation, the pretraining pool must exclude downstream validation and test patients.

## Run downstream fine-tuning

The following commands invoke the current script. They are not evidence that the implementation issues have been resolved.

Linux or macOS:

```bash
export DATA_ROOT='./data/train'
export MAE_PRETRAINED_PATH='./weights/mae_tradition_mit.pth'
export MODEL_SAVE_PATH='./weights/best_model_weight.pth'
python train.py
```

Windows PowerShell:

```powershell
$env:DATA_ROOT = './data/train'
$env:MAE_PRETRAINED_PATH = './weights/mae_tradition_mit.pth'
$env:MODEL_SAVE_PATH = './weights/best_model_weight.pth'
python train.py
```

The defaults are 50 epochs, batch size 20, input size 512, and seed 42. The encoder learning rate is `1e-4` and the head learning rate is `3e-4`. These values are constants in `train.py`; setting an environment variable named `BATCH_SIZE` does not change training batch size.

The script randomly reserves 20% of **samples** for validation. It does not load a patient split manifest, accept a separate validation root, or export its split. Do not describe this default as patient-disjoint evaluation. A missing MAE checkpoint causes fallback to ImageNet initialization rather than an error, so confirm the loading message when conducting the MAE ablation.

## Run inference

Set `MODEL_PATH` explicitly because its default differs from the training checkpoint destination.

```bash
export DATA_ROOT='./data/val'
export MODEL_PATH='./weights/best_model_weight.pth'
export OUTPUT_DIR='./predictions'
export BATCH_SIZE='8'
python model.py
```

```powershell
$env:DATA_ROOT = './data/val'
$env:MODEL_PATH = './weights/best_model_weight.pth'
$env:OUTPUT_DIR = './predictions'
$env:BATCH_SIZE = '8'
python model.py
```

Inference builds heads from task IDs present in its input CSVs and loads a plain full-model state dictionary strictly. The head set and dimensions must match the checkpoint. A subset of tasks can therefore fail to load a checkpoint trained with all 27 heads. Do not suppress checkpoint mismatches without checking the model definition.

For task types present in the input, inference exports segmentation masks and/or `classification_predictions.json`, `detection_predictions.json`, and `regression_predictions.json`. Coordinates are converted back to original image dimensions. Use a new output directory for each experiment to avoid mixing old and new predictions.

## Evaluate exported predictions

`evaluate.py` has hard-coded paths in its main block. Either edit that block or use its Python interface. The following invocation uses the existing interface without modifying the script:

```bash
python -c "from pathlib import Path; from evaluate import Evaluator; Path('evaluation').mkdir(exist_ok=True); e=Evaluator('./data/val','./predictions'); r=e.evaluate_all(); e.print_summary(r,save_path='evaluation/summary.txt'); e.save_results(r,'evaluation/results.json')"
```

This command also works in PowerShell. It requires ground-truth labels and the matching exported predictions. The inference CSV paths must match the evaluation CSV paths exactly for JSON lookup.

**Two evaluation paths use different definitions.** Internal validation reports coordinate MAE using a default 256-by-256 reference grid. Standalone evaluation reports landmark MRE in original-image pixels. Segmentation and classification aggregation also differ. Neither the internal checkpoint-selection score nor the standalone summary is an implemented official normalized challenge score. Read [Evaluation](docs/EVALUATION.md) before comparing results with a paper or a challenge leaderboard.

## Reproduction and availability

Use [Reproducibility](docs/REPRODUCIBILITY.md) to record the configuration and [Known issues](docs/KNOWN_ISSUES.md) to identify changes that could alter results. No paper performance values are supplied because they have not been traced to this exact source snapshot, split, checkpoint, and evaluator.

The supplied files contain no LICENSE file or verified publication metadata. This documentation does not assign a license or invent a citation. See [Code availability](CODE_AVAILABILITY.md) and the [release checklist](docs/RELEASE_CHECKLIST.md) for information that the authors need to finalize.

This is research software. It has not been validated for patient care.
