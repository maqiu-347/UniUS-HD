# Utility functions for training, validation, and evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from model_factory import TASK_CONFIGURATIONS  # Needed for task name mapping


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def multi_task_collate_fn(batch):
    """
    Custom collate function to handle different label shapes in multi-task learning.
    Images are stacked; labels and task_ids remain as lists.
    """
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    task_ids = [item['task_id'] for item in batch]

    # Stack images as they have consistent dimensions
    images = torch.stack(images, 0)

    return {'image': images, 'label': labels, 'task_id': task_ids}


class DetectionLoss(nn.Module):
    """Faster R-CNN style detection loss with IoU-based target assignment."""
    def __init__(self, classification_weight=1.0, box_regression_weight=5.0,
                 iou_positive_thresh=0.5, iou_negative_thresh=0.3):
        super().__init__()
        self.classification_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.box_regression_loss = nn.SmoothL1Loss(reduction='none', beta=1.0 / 9.0)
        self.cls_w = classification_weight
        self.box_w = box_regression_weight
        self.iou_pos_thresh = iou_positive_thresh
        self.iou_neg_thresh = iou_negative_thresh

    def compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-6)

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 5, H, W] - first 4 channels are bbox, 5th is confidence
            targets: [B, 4] - GT boxes (normalized coordinates)
        Returns:
            total_loss: classification loss + regression loss
        """
        batch_size, _, h, w = predictions.shape
        device = predictions.device

        total_cls_loss = 0.0
        total_box_loss = 0.0
        num_valid = 0

        for b in range(batch_size):
            pred = predictions[b]
            target = targets[b]

            # Skip invalid targets
            if target[0] < 0:
                continue

            num_valid += 1

            pred_boxes = pred[:4].permute(1, 2, 0).reshape(-1, 4)
            pred_scores = pred[4].reshape(-1)

            target_expanded = target.unsqueeze(0)
            ious = self.compute_iou_matrix(pred_boxes, target_expanded).squeeze(-1)

            pos_mask = ious >= self.iou_pos_thresh
            neg_mask = ious < self.iou_neg_thresh

            # If no positive sample, take the one with the highest IoU as positive
            if not pos_mask.any():
                best_idx = ious.argmax()
                pos_mask[best_idx] = True
                neg_mask[best_idx] = False

            cls_targets = torch.zeros_like(pred_scores)
            cls_targets[pos_mask] = 1.0

            # Only use positive and negative samples for classification loss
            valid_mask = pos_mask | neg_mask
            if valid_mask.any():
                cls_loss = self.classification_loss(pred_scores[valid_mask], cls_targets[valid_mask])
                total_cls_loss += cls_loss.mean()

            # Regression loss only on positive samples
            if pos_mask.any():
                box_loss = self.box_regression_loss(pred_boxes[pos_mask], target.expand(pos_mask.sum(), -1))
                total_box_loss += box_loss.mean()

        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_cls_loss /= num_valid
        total_box_loss /= num_valid

        return self.cls_w * total_cls_loss + self.box_w * total_box_loss


class SimpleDetectionLoss(nn.Module):
    """Original simple detection loss (kept as fallback)."""
    def __init__(self, classification_weight=1.0, box_regression_weight=8.0):
        super().__init__()
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.box_regression_loss = nn.L1Loss()
        self.cls_w, self.box_w = classification_weight, box_regression_weight

    def forward(self, predictions, targets):
        pred_boxes, pred_scores = predictions[:, :4], predictions[:, 4].squeeze(-1)
        valid_indices = (targets[:, 0] >= 0).view(-1)
        if not valid_indices.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        cls_loss = self.classification_loss(pred_scores[valid_indices], torch.ones_like(pred_scores)[valid_indices])
        box_loss = self.box_regression_loss(pred_boxes[valid_indices], targets[valid_indices])
        return self.cls_w * cls_loss + self.box_w * box_loss


def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return accuracy_score(y_true, y_pred)


def calculate_f1_score(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def calculate_dice_coefficient(y_true, y_pred_logits):
    y_pred_mask = torch.argmax(y_pred_logits, dim=1)
    num_classes = y_pred_logits.shape[1]
    y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2)
    y_pred_one_hot = F.one_hot(y_pred_mask, num_classes=num_classes).permute(0, 3, 1, 2)
    intersection = torch.sum(y_true_one_hot[:, 1:] * y_pred_one_hot[:, 1:])
    union = torch.sum(y_true_one_hot[:, 1:]) + torch.sum(y_pred_one_hot[:, 1:])
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()


def calculate_mae(y_true, y_pred, image_size=(256, 256)):
    h, w = image_size
    y_true_px = y_true.cpu().numpy().copy()
    y_pred_px = y_pred.cpu().numpy().copy()
    y_true_px[:, 0::2] *= w
    y_true_px[:, 1::2] *= h
    y_pred_px[:, 0::2] *= w
    y_pred_px[:, 1::2] *= h
    return np.mean(np.abs(y_true_px - y_pred_px))


def calculate_iou(y_true, y_pred):
    """Compute IoU; supports Faster R-CNN style outputs."""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    batch_ious = []
    for i in range(y_true.shape[0]):
        box_true, box_pred = y_true[i], y_pred[i]
        xA = max(box_true[0], box_pred[0])
        yA = max(box_true[1], box_pred[1])
        xB = min(box_true[2], box_pred[2])
        yB = min(box_true[3], box_pred[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box_true_area = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
        box_pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        union_area = box_true_area + box_pred_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        batch_ious.append(iou)
    return np.mean(batch_ious)


def extract_best_detection(outputs, method='max_score'):
    """
    Extract the best detection box from Faster R-CNN style grid outputs.

    Args:
        outputs: [B, 5, H, W] - first 4 channels are bbox, 5th is confidence
        method: 'max_score' or 'center'

    Returns:
        final_boxes: [B, 4]
        scores: [B]
    """
    batch_size, _, h, w = outputs.shape
    device = outputs.device

    final_boxes = torch.zeros((batch_size, 4), device=device)
    scores = torch.zeros(batch_size, device=device)

    for i in range(batch_size):
        pred = outputs[i]
        pred_scores = pred[4]
        pred_boxes = pred[:4]

        if method == 'max_score':
            flat_scores = pred_scores.view(-1)
            best_idx = flat_scores.argmax()
            best_h = best_idx // w
            best_w = best_idx % w
        else:  # center
            best_h = h // 2
            best_w = w // 2

        final_boxes[i] = pred_boxes[:, best_h, best_w]
        scores[i] = torch.sigmoid(pred_scores[best_h, best_w])

    return final_boxes, scores


def evaluate(model, val_loader, device):
    """
    Evaluation loop supporting multi-task batches.
    """
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Validation]")
        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label']
            task_ids = batch['task_id']
            unique_tasks_in_batch = set(task_ids)

            for task_id in unique_tasks_in_batch:
                task_name = task_id_to_name[task_id]
                task_indices = [i for i, t_id in enumerate(task_ids) if t_id == task_id]
                task_images = images[task_indices]
                task_labels_list = [labels[i] for i in task_indices]
                task_labels = torch.stack(task_labels_list, 0)

                outputs = model(task_images, task_id=task_id)

                if task_name == 'classification':
                    task_metrics[task_id]['Accuracy'].append(calculate_accuracy(task_labels, outputs))
                    task_metrics[task_id]['F1-Score'].append(calculate_f1_score(task_labels, outputs))

                elif task_name == 'segmentation':
                    task_metrics[task_id]['Dice'].append(calculate_dice_coefficient(task_labels.to(device), outputs))

                elif task_name == 'Regression':
                    task_metrics[task_id]['MAE (pixels)'].append(calculate_mae(task_labels, outputs))

                elif task_name == 'detection':
                    # Two-stage Faster R-CNN: outputs is list[{'boxes','scores'}]
                    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
                        final_boxes = torch.zeros((len(outputs), 4), device=device)
                        det_scores = torch.zeros((len(outputs),), device=device)
                        for i, out in enumerate(outputs):
                            boxes = out.get('boxes', torch.zeros(4, device=device))
                            scores = out.get('scores', torch.tensor(0.0, device=device))
                            if not isinstance(boxes, torch.Tensor):
                                boxes = torch.tensor(boxes, device=device)
                            if not isinstance(scores, torch.Tensor):
                                scores = torch.tensor(scores, device=device)
                            final_boxes[i] = boxes.to(device)
                            det_scores[i] = scores.to(device).reshape(())
                    else:
                        # Legacy grid output: [B, 5, H, W]
                        final_boxes, det_scores = extract_best_detection(outputs, method='max_score')

                    task_metrics[task_id]['IoU'].append(calculate_iou(task_labels, final_boxes))
                    task_metrics[task_id]['Avg_Confidence'].append(det_scores.mean().item())

    results = []
    sorted_task_ids = sorted(list(task_id_to_name.keys()))
    for task_id in sorted_task_ids:
        if task_id in task_metrics:
            task_name = task_id_to_name[task_id]
            result_row = {'Task ID': task_id, 'Task Name': task_name}
            for metric_name, values in task_metrics[task_id].items():
                result_row[metric_name] = np.mean(values)
            results.append(result_row)
    return pd.DataFrame(results)