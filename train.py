import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random
import os

# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn,
    evaluate,
    DetectionLoss,
    set_seed
)

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 20
NUM_EPOCHS = 50
DATA_ROOT_PATH = os.environ.get('DATA_ROOT', './data/train')
ENCODER = 'mit_b5'
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = os.environ.get('MODEL_SAVE_PATH', './weights/best_model_weight.pth')
VAL_SPLIT = 0.2
INPUT_SIZE = 512


def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Training transforms with augmentation
    train_transforms = A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # Validation transforms without augmentation
    val_transforms = A.Compose([
        A.Resize(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # Load dataset and split
    temp_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    dataset_size = len(temp_dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = list(range(dataset_size))
    train_indices, val_indices = torch.utils.data.random_split(
        indices, [train_size, val_size], generator=generator
    )

    train_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    val_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=val_transforms)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")

    # Fix dataframe reference for subset (used by the balanced sampler)
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices.indices].reset_index(drop=True)

    # Build balanced multi-task sampler for training
    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=BATCH_SIZE)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=multi_task_collate_fn,
    )

    # Build model
    model = MultiTaskModelFactory(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        task_configs=TASK_CONFIGURATIONS
    ).to(device)

    # Load MAE pretrained weights if available
    MAE_PRETRAINED_PATH = os.environ.get('MAE_PRETRAINED_PATH', './weights/mae_tradition_mit.pth')
    if os.path.exists(MAE_PRETRAINED_PATH):
        print("Loading MAE pretrained weights...")
        ckpt = torch.load(MAE_PRETRAINED_PATH, map_location=device)
        # Support both raw state_dict and dict containing 'encoder'
        if isinstance(ckpt, dict) and 'encoder' in ckpt:
            mae_weights = ckpt['encoder']
        else:
            mae_weights = ckpt
        model.encoder.load_state_dict(mae_weights)
        print("MAE weights loaded successfully!")
    else:
        print("No MAE pretrained weights found, using ImageNet weights")

    # Loss functions
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'),
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(),
        'detection': DetectionLoss()
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # Optimizer with per-group learning rates
    print("\n--- Setting parameter groups ---")
    param_groups = [{'params': model.encoder.parameters(), 'lr': LEARNING_RATE * 1}]
    print(f"  - Shared Encoder  -> LR: {LEARNING_RATE * 1}")

    for task_id, head in model.heads.items():
        lr_multiplier = 3.0
        current_lr = LEARNING_RATE * lr_multiplier
        param_groups.append({'params': head.parameters(), 'lr': current_lr})
        print(f"  - Task Head '{task_id:<25}' -> LR: {current_lr}")

    optimizer = optim.AdamW(param_groups, weight_decay=0.01)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    print("\n--- Cosine Annealing Scheduler configured ---")

    best_val_score = -float('inf')
    print("\n" + "=" * 50 + "\n--- Start Training ---")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")

        for batch in loop:
            images = batch['image'].to(device)
            task_ids = batch['task_id']

            # Ensure all samples in the batch belong to the same task
            unique_tasks = set(task_ids)
            if len(unique_tasks) > 1:
                print(f"[WARNING] Mixed tasks in batch: {unique_tasks}")
                continue

            labels = torch.stack(batch['label']).to(device)

            # All samples in batch belong to the same task due to sampler
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            # Detection needs targets to compute two-stage loss
            if task_name == 'detection':
                outputs = model(images, task_id=current_task_id, targets=labels)
                if isinstance(outputs, dict) and 'total_loss' in outputs:
                    loss = outputs['total_loss']
                else:
                    raise RuntimeError("Detection head did not return losses dict during training")
            else:
                outputs = model(images, task_id=current_task_id)
                loss = loss_functions[task_name](outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_losses[current_task_id].append(loss.item())
            current_lr = scheduler.get_last_lr()[0]
            loop.set_postfix(
                loss=loss.item(),
                task=current_task_id,
                lr=f"{current_lr:.2e}"
            )

        # Report average training loss per task
        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
        print("-" * 40)

        # Validation
        val_results_df = evaluate(model, val_loader, device)

        score_cols = [col for col in val_results_df.columns
                      if 'MAE' not in col and
                      isinstance(val_results_df[col].iloc[0], (int, float))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        # Save best model
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            save_dir = os.path.dirname(MODEL_SAVE_PATH)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved! Score improved to: {best_val_score:.4f}\n")

        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()