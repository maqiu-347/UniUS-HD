# Sample Code Submission for Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional
import json
from model_factory import MultiTaskModelFactory


class InferenceDataset(Dataset):
    """Inference dataset class"""

    def __init__(self, data_root: str, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.csv_path = os.path.join(self.data_root, 'csv_files')

        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")

        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")

        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        print(f"Data loading complete. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']

        # Load image
        image_rel_path = record['image_path']
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, image_rel_path))
        image = cv2.imread(image_abs_path)

        # Robustness check: retry next index if image load fails
        if image is None:
            print(f"Warning: Unable to load image {image_abs_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        # Get mask_path (if segmentation task)
        mask_path = None
        if task_name == 'segmentation' and 'mask_path' in record and pd.notna(record['mask_path']):
            mask_path = record['mask_path']

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        # Return data including metadata
        return {
            'image': image,
            'task_id': task_id,
            'task_name': task_name,
            'image_path': image_rel_path,
            'mask_path': mask_path,
            'original_size': (original_height, original_width),
            'index': idx
        }


def inference_collate_fn(batch):
    """Inference collate function that preserves metadata"""
    images = torch.stack([item['image'] for item in batch], 0)
    task_ids = [item['task_id'] for item in batch]
    task_names = [item['task_name'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    indices = [item['index'] for item in batch]

    return {
        'image': images,
        'task_id': task_ids,
        'task_name': task_names,
        'image_path': image_paths,
        'mask_path': mask_paths,
        'original_size': original_sizes,
        'index': indices
    }


class Model:
    """
    Foundation Model for Ultrasound Image Analysis
    Supports four task types: segmentation, classification, Regression, detection
    """

    @staticmethod
    def convert_to_python_type(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: Model.convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Model.convert_to_python_type(i) for i in obj]
        else:
            return obj

    def __init__(self):
        """Initialize model and load pretrained weights"""
        print("Initializing model...")

        # Set compute device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device use: {self.device}")

        # Model variables, will be initialized in predict()
        self.model = None
        self.task_configs = None
        self.task_id_to_name = None

        # Define data preprocessing transforms (no augmentation for inference)
        INPUT_SIZE = 512
        self.transforms = A.Compose([
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        print("Initializing model is done!\n")

    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        """
        Perform prediction on input data

        Args:
            data_root: Data root directory containing csv_files subdirectory
            output_dir: Output results root directory
            batch_size: Batch size, default is 8

        Output:
            - Segmentation tasks: Save predicted masks as image files
            - Classification tasks: Save to classification_predictions.json
            - Detection tasks: Save to detection_predictions.json
            - Regression tasks: Save to regression_predictions.json
        """
        print(f"{'='*60}")
        print(f"Starting prediction...")
        print(f"Data directory: {data_root}")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")

        os.makedirs(output_dir, exist_ok=True)

        # Load dataset
        print(f"Loading dataset...")
        dataset = InferenceDataset(data_root=data_root, transforms=self.transforms)

        # Build task configurations from dataset
        print(f"\nBuilding task configurations from dataset...")
        self.task_configs = []
        task_config_map = {}

        for _, row in dataset.dataframe.iterrows():
            task_id = row['task_id']
            if task_id not in task_config_map:
                task_config = {
                    'task_id': task_id,
                    'task_name': row['task_name'],
                    'num_classes': int(row['num_classes'])
                }
                task_config_map[task_id] = task_config
                self.task_configs.append(task_config)

        print(f"Detected {len(self.task_configs)} task configurations")
        for cfg in sorted(self.task_configs, key=lambda x: x['task_id']):
            print(f"  - {cfg['task_id']}: {cfg['task_name']}, num_classes={cfg['num_classes']}")

        self.task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in self.task_configs}

        # Create and load model
        print(f"\nLoading model...")
        self.model = MultiTaskModelFactory(
            encoder_name='mit_b5',
            encoder_weights='imagenet',
            task_configs=self.task_configs
        ).to(self.device)

        # Load trained model weights
        model_path = os.environ.get('MODEL_PATH', './best_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Model weights loaded successfully!")

        # Create data loader (batch processing for faster inference)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=inference_collate_fn
        )

        # Batch inference
        print(f"\nStarting inference...")
        classification_results = []
        detection_results = []
        regression_results = []
        task_counts = {}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction progress"):
                images = batch['image'].to(self.device)
                task_ids = batch['task_id']
                task_names = batch['task_name']
                image_paths = batch['image_path']
                mask_paths = batch['mask_path']
                original_sizes = batch['original_size']

                # Process each task in batch
                unique_tasks = list(set(task_ids))

                for task_id in unique_tasks:
                    # Get indices for all samples of current task
                    task_indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
                    task_images = images[task_indices]

                    # Model inference
                    outputs = self.model(task_images, task_id=task_id)
                    task_name = task_names[task_indices[0]]

                    # Save prediction results for each sample
                    for i, batch_idx in enumerate(task_indices):
                        pred = outputs[i]
                        image_path = image_paths[batch_idx]
                        mask_path = mask_paths[batch_idx]
                        original_size = original_sizes[batch_idx]

                        task_counts[task_id] = task_counts.get(task_id, 0) + 1

                        # Process results by task type
                        if task_name == 'segmentation':
                            self._save_segmentation(pred, image_path, mask_path, output_dir, original_size)
                        elif task_name == 'classification':
                            result = self._process_classification(pred, task_id, image_path)
                            classification_results.append(result)
                        elif task_name == 'Regression':
                            result = self._process_regression(pred, task_id, image_path, original_size)
                            regression_results.append(result)
                        elif task_name == 'detection':
                            result = self._process_detection(pred, task_id, image_path, original_size)
                            detection_results.append(result)

        # Save aggregated JSON results
        print("\nSaving prediction results...")

        if classification_results:
            json_path = os.path.join(output_dir, 'classification_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(classification_results, f, indent=2, ensure_ascii=False)
            print(f"  - Classification results: {json_path} ({len(classification_results)} samples)")

        if detection_results:
            json_path = os.path.join(output_dir, 'detection_predictions.json')
            detection_results = self.convert_to_python_type(detection_results)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_results, f, indent=2, ensure_ascii=False)
            print(f"  - Detection results: {json_path} ({len(detection_results)} samples)")

        if regression_results:
            json_path = os.path.join(output_dir, 'regression_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(regression_results, f, indent=2, ensure_ascii=False)
            print(f"  - Regression results: {json_path} ({len(regression_results)} samples)")

        # Print statistics
        print(f"\n{'='*60}")
        print("Prediction complete!")
        print(f"{'='*60}")
        print("\nPrediction count by task:")
        for task_id in sorted(task_counts.keys()):
            task_name_str = self.task_id_to_name.get(task_id, 'unknown')
            count = task_counts[task_id]
            print(f"  - {task_id:<25} ({task_name_str:<15}): {count:>5} samples")
        print(f"\nTotal: {sum(task_counts.values())} samples")
        print()

    def _save_segmentation(self, pred, image_path, mask_path, output_dir, original_size):
        """
        Save segmentation prediction results as image file

        Args:
            pred: Prediction result (C, H, W) or (H, W)
            image_path: Image path
            mask_path: Mask path (from CSV)
            output_dir: Output root directory
            original_size: Original image size (height, width)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # For multi-class segmentation (C, H, W), take argmax
        if pred.ndim == 3:
            mask = np.argmax(pred, axis=0).astype(np.uint8)
        else:
            mask = pred.astype(np.uint8)

        # Resize back to original size
        h, w = original_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Determine output path
        if mask_path:
            # Use mask path specified in CSV, remove leading '../'
            mask_path_clean = mask_path.replace('../', '')
            output_path = os.path.join(output_dir, mask_path_clean)
        else:
            # Default: replace keywords in image_path
            default_mask_path = image_path.replace('img', 'mask').replace('IMG', 'MASK')
            output_path = os.path.join(output_dir, default_mask_path)

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, mask)

    def _process_classification(self, pred, task_id, image_path):
        """
        Process classification task prediction results

        Args:
            pred: Prediction logits (num_classes,)
            task_id: Task ID
            image_path: Image path

        Returns:
            Dictionary containing prediction results
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # Get predicted class
        pred_class = int(np.argmax(pred))

        # Stable softmax to avoid numerical overflow
        pred_exp = np.exp(pred - np.max(pred))
        pred_probs = pred_exp / np.sum(pred_exp)

        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_class': pred_class,
            'predicted_probs': pred_probs.tolist()
        }

    def _process_regression(self, pred, task_id, image_path, original_size):
        """
        Process regression task prediction results (keypoint localization)

        Args:
            pred: Predicted coordinates (num_points * 2,) - normalized coordinates
            task_id: Task ID
            image_path: Image path
            original_size: Original image size (height, width)

        Returns:
            Dictionary containing prediction results
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        # Normalized coordinates
        coords = pred.flatten().tolist()

        # Convert to pixel coordinates
        h, w = original_size
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i + 1]
            x_pixel = x_norm * w
            y_pixel = y_norm * h
            pixel_coords.extend([x_pixel, y_pixel])

        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_points_normalized': coords,
            'predicted_points_pixels': pixel_coords
        }

    def _process_detection(self, pred, task_id, image_path, original_size):
        """
        Process detection task prediction results

        Args:
            pred:
              - two-stage output: dict with 'boxes' and 'scores'
              - legacy output: (5, H, W) grid-based predictions
            task_id: Task ID
            image_path: Image path
            original_size: Original image size (height, width)

        Returns:
            Dictionary containing prediction results
        """
        # Two-stage Faster R-CNN: pred is dict {'boxes': Tensor[4], 'scores': Tensor[]}
        if isinstance(pred, dict):
            boxes = pred.get('boxes', None)
            score = pred.get('scores', None)

            if isinstance(boxes, torch.Tensor):
                bbox_norm = boxes.detach().cpu().flatten().numpy()
            elif boxes is None:
                bbox_norm = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                bbox_norm = np.array(boxes, dtype=np.float32).flatten()

            if isinstance(score, torch.Tensor):
                best_score = float(score.detach().cpu().reshape(()).item())
            elif score is None:
                best_score = 0.0
            else:
                best_score = float(score)

            bbox_norm_list = bbox_norm.tolist()
        else:
            # Legacy grid output: (5, H, W)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()

            _, h, w = pred.shape

            # Find location with highest confidence score
            scores = pred[4, :, :].flatten()
            scores_prob = 1 / (1 + np.exp(-scores))

            best_idx = np.argmax(scores_prob)
            best_h = best_idx // w
            best_w = best_idx % w
            best_score = float(scores_prob[best_idx])

            # Extract predicted bbox at that location
            bbox_norm = pred[:4, best_h, best_w]
            bbox_norm_list = bbox_norm.tolist()

        # Convert to pixel coordinates
        img_h, img_w = original_size
        bbox_pixel = [
            bbox_norm_list[0] * img_w,
            bbox_norm_list[1] * img_h,
            bbox_norm_list[2] * img_w,
            bbox_norm_list[3] * img_h,
        ]

        return {
            'image_path': image_path,
            'task_id': task_id,
            'bbox_normalized': bbox_norm_list,
            'bbox_pixels': bbox_pixel,
            'confidence': float(best_score)
        }


if __name__ == '__main__':
    """
    Usage example.
    """
    data_root = os.environ.get('DATA_ROOT', './data/val')
    output_dir = os.environ.get('OUTPUT_DIR', './predictions')
    batch_size = int(os.environ.get('BATCH_SIZE', '8'))

    model = Model()
    model.predict(data_root, output_dir, batch_size=batch_size)

    print("Inference complete!")