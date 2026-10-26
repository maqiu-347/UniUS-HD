"""
Faster R-CNN detection head adapted to a single-scale FPN feature.
- Stage 1: RPN generates proposals.
- Stage 2: RoI Align + RoI heads classify and refine proposals.
Training : returns a dict of losses including 'total_loss'.
Inference: returns list[{'boxes': Tensor[4], 'scores': Tensor[]}].
"""

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from torchvision.ops import roi_align, nms


class AnchorGenerator(nn.Module):
    """
    Multi-scale / multi-aspect-ratio anchor generator.
    A single FPN level is used here, so `image_size` is required to
    compute the stride.
    """
    def __init__(
        self,
        sizes: Tuple[int, ...] = (16, 32, 64, 128, 256),
        aspect_ratios: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
    ):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build base anchors centered at (0, 0)."""
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, device: torch.device, dtype: torch.dtype):
        self.cell_anchors = self.generate_anchors(
            list(self.sizes), list(self.aspect_ratios), device, dtype
        )

    @property
    def num_anchors_per_location(self) -> int:
        return len(self.sizes) * len(self.aspect_ratios)

    def grid_anchors(
        self, grid_sizes: List[Tuple[int, int]], strides: List[Tuple[int, int]], device: torch.device
    ) -> List[torch.Tensor]:
        """Place base anchors on every spatial location of the feature map."""
        anchors = []
        cell_anchors = self.cell_anchors

        for size, stride in zip(grid_sizes, strides):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + cell_anchors.view(1, -1, 4)).reshape(-1, 4)
            )
        return anchors

    def forward(self, feature_maps: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        device = feature_maps.device
        dtype = feature_maps.dtype

        if self.cell_anchors is None:
            self.set_cell_anchors(device, dtype)

        grid_size = (feature_maps.shape[2], feature_maps.shape[3])
        # Integer stride between image and feature map
        stride = (
            image_size[0] // grid_size[0],
            image_size[1] // grid_size[1]
        )

        anchors = self.grid_anchors([grid_size], [stride], device)
        return anchors[0]


class RPNHead(nn.Module):
    """Region Proposal Network head (objectness + bbox regression)."""
    def __init__(self, in_channels: int, num_anchors: int, num_conv: int = 2):
        super().__init__()
        conv_layers: List[nn.Module] = []
        for _ in range(num_conv):
            conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.BatchNorm2d(in_channels))
            conv_layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv_layers)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.conv(x)
        logits = self.cls_logits(t)
        bbox_reg = self.bbox_pred(t)
        return logits, bbox_reg


class BoxCoder:
    """Encode / decode boxes relative to proposals (standard R-CNN parameterization)."""
    def __init__(self, weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)):
        self.weights = weights

    def encode(self, reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        wx, wy, ww, wh = self.weights

        proposals_x1 = proposals[:, 0].unsqueeze(1)
        proposals_y1 = proposals[:, 1].unsqueeze(1)
        proposals_x2 = proposals[:, 2].unsqueeze(1)
        proposals_y2 = proposals[:, 3].unsqueeze(1)

        reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
        reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
        reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
        reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

        ex_widths = proposals_x2 - proposals_x1
        ex_heights = proposals_y2 - proposals_y1
        ex_ctr_x = proposals_x1 + 0.5 * ex_widths
        ex_ctr_y = proposals_y1 + 0.5 * ex_heights

        gt_widths = reference_boxes_x2 - reference_boxes_x1
        gt_heights = reference_boxes_y2 - reference_boxes_y1
        gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
        gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-6)
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-6)
        targets_dw = ww * torch.log(gt_widths / (ex_widths + 1e-6) + 1e-6)
        targets_dh = wh * torch.log(gt_heights / (ex_heights + 1e-6) + 1e-6)

        targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Clip to avoid exp overflow
        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


class RoIHeads(nn.Module):
    """Fast R-CNN RoI head: RoI Align -> FC -> class + bbox."""
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        roi_output_size: int = 7,
        representation_size: int = 1024,
    ):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.in_channels = in_channels

        self.fc6 = nn.Linear(in_channels * roi_output_size * roi_output_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

        # num_classes + 1 because class 0 is background
        self.cls_score = nn.Linear(representation_size, num_classes + 1)
        self.bbox_pred = nn.Linear(representation_size, 4)

        self.dropout = nn.Dropout(0.5)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(
        self,
        features: torch.Tensor,
        proposals: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if proposals.numel() == 0:
            device = features.device
            return (
                torch.zeros((0, 2), device=device),
                torch.zeros((0, 4), device=device),
            )

        spatial_scale = features.shape[2] / image_size[0]

        box_features = roi_align(
            features,
            [proposals],
            output_size=self.roi_output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=2,
        )

        box_features = box_features.flatten(start_dim=1)
        box_features = F.relu(self.fc6(box_features))
        box_features = self.dropout(box_features)
        box_features = F.relu(self.fc7(box_features))
        box_features = self.dropout(box_features)

        class_logits = self.cls_score(box_features)
        box_regression = self.bbox_pred(box_features)

        return class_logits, box_regression


class TwoStageDetectionHead(nn.Module):
    """
    Full two-stage Faster R-CNN head.
    Training : returns {'rpn_objectness_loss', 'rpn_box_loss',
                        'roi_cls_loss', 'roi_box_loss', 'total_loss'}
    Inference: returns list[{'boxes', 'scores'}] (one best box per image).
    """

    def __init__(
        self,
        fpn_out_channels: int,
        num_classes: int = 1,
        image_size: Tuple[int, int] = (512, 512),
        anchor_sizes: Tuple[int, ...] = (16, 32, 64, 128, 256),
        anchor_aspect_ratios: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 1000,
        rpn_post_nms_top_n_test: int = 300,
        rpn_nms_thresh: float = 0.7,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        box_roi_pool_size: int = 7,
        box_head_representation_size: int = 1024,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 128,
        box_positive_fraction: float = 0.25,
        box_nms_thresh: float = 0.5,
        box_score_thresh: float = 0.05,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = len(anchor_sizes) * len(anchor_aspect_ratios)
        # Kept as a parameter so the same head works at any input resolution
        self.image_size = image_size

        self.rpn_pre_nms_top_n = {'training': rpn_pre_nms_top_n_train, 'testing': rpn_pre_nms_top_n_test}
        self.rpn_post_nms_top_n = {'training': rpn_post_nms_top_n_train, 'testing': rpn_post_nms_top_n_test}
        self.rpn_nms_thresh = rpn_nms_thresh
        self.rpn_fg_iou_thresh = rpn_fg_iou_thresh
        self.rpn_bg_iou_thresh = rpn_bg_iou_thresh
        self.rpn_batch_size_per_image = rpn_batch_size_per_image
        self.rpn_positive_fraction = rpn_positive_fraction

        self.box_fg_iou_thresh = box_fg_iou_thresh
        self.box_bg_iou_thresh = box_bg_iou_thresh
        self.box_batch_size_per_image = box_batch_size_per_image
        self.box_positive_fraction = box_positive_fraction
        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh

        self.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=anchor_aspect_ratios)
        self.rpn_head = RPNHead(fpn_out_channels, self.num_anchors, num_conv=2)
        self.roi_heads = RoIHeads(
            in_channels=fpn_out_channels,
            num_classes=num_classes,
            roi_output_size=box_roi_pool_size,
            representation_size=box_head_representation_size,
        )
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

    def compute_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-6)

    def filter_proposals(
        self,
        proposals: torch.Tensor,
        objectness: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Top-k + clip + remove small + NMS for RPN proposals."""
        batch_size = proposals.shape[0]
        objectness = objectness.reshape(batch_size, -1)
        proposals = proposals.reshape(batch_size, -1, 4)

        all_proposals: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []

        for b in range(batch_size):
            obj = objectness[b]
            prop = proposals[b]

            pre_nms_top_n = self.rpn_pre_nms_top_n['training'] if self.training else self.rpn_pre_nms_top_n['testing']
            pre_nms_top_n = min(pre_nms_top_n, obj.shape[0])
            _, top_n_idx = obj.topk(pre_nms_top_n)

            obj = obj[top_n_idx]
            prop = prop[top_n_idx]

            prop = box_ops.clip_boxes_to_image(prop, image_size)

            keep = box_ops.remove_small_boxes(prop, min_size=1e-3)
            prop = prop[keep]
            obj = obj[keep]

            keep = nms(prop, obj, self.rpn_nms_thresh)
            post_nms_top_n = self.rpn_post_nms_top_n['training'] if self.training else self.rpn_post_nms_top_n['testing']
            keep = keep[:post_nms_top_n]

            prop = prop[keep]
            obj = obj[keep]

            all_proposals.append(prop)
            all_scores.append(obj)

        return all_proposals, all_scores

    def assign_targets_to_anchors(
        self, anchors: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign RPN labels: 1 positive, 0 negative, -1 ignore."""
        device = anchors.device
        num_anchors = anchors.shape[0]

        if targets.numel() == 0:
            labels = torch.zeros(num_anchors, dtype=torch.float32, device=device)
            matched_gt_boxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
            return labels, matched_gt_boxes

        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        iou_matrix = self.compute_iou(anchors, targets)
        matched_vals, matches = iou_matrix.max(dim=1)

        labels = torch.zeros(num_anchors, dtype=torch.float32, device=device)
        labels[matched_vals >= self.rpn_fg_iou_thresh] = 1.0
        labels[matched_vals < self.rpn_bg_iou_thresh] = 0.0
        labels[(matched_vals >= self.rpn_bg_iou_thresh) & (matched_vals < self.rpn_fg_iou_thresh)] = -1.0

        # Each GT box must have at least one positive anchor
        _, gt_argmax = iou_matrix.max(dim=0)
        labels[gt_argmax] = 1.0

        matched_gt_boxes = targets[matches]
        return labels, matched_gt_boxes

    def sample_proposals(
        self, proposals: torch.Tensor, targets: torch.Tensor, image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample 128 proposals with 1:3 positive:negative ratio."""
        device = proposals.device

        if proposals.numel() == 0:
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros((0, 4), device=device),
            )

        # GT boxes are normalized; convert to pixel coordinates
        gt_boxes = targets * torch.tensor(
            [image_size[1], image_size[0], image_size[1], image_size[0]],
            device=device,
            dtype=torch.float32,
        )
        if gt_boxes.dim() == 1:
            gt_boxes = gt_boxes.unsqueeze(0)

        iou_matrix = self.compute_iou(proposals, gt_boxes)
        matched_vals, matches = iou_matrix.max(dim=1)

        labels = torch.zeros(proposals.shape[0], dtype=torch.long, device=device)
        labels[matched_vals >= self.box_fg_iou_thresh] = 1

        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (matched_vals < self.box_bg_iou_thresh).nonzero(as_tuple=True)[0]

        num_pos = int(self.box_batch_size_per_image * self.box_positive_fraction)
        num_pos = min(num_pos, pos_idx.numel())
        num_neg = self.box_batch_size_per_image - num_pos
        num_neg = min(num_neg, neg_idx.numel())

        if num_pos > 0:
            perm_pos = torch.randperm(pos_idx.numel(), device=device)[:num_pos]
            pos_idx = pos_idx[perm_pos]
        else:
            pos_idx = pos_idx[:0]

        if num_neg > 0:
            perm_neg = torch.randperm(neg_idx.numel(), device=device)[:num_neg]
            neg_idx = neg_idx[perm_neg]
        else:
            neg_idx = neg_idx[:0]

        sampled_idx = torch.cat([pos_idx, neg_idx])
        sampled_proposals = proposals[sampled_idx]
        sampled_labels = labels[sampled_idx]
        sampled_gt_boxes = gt_boxes[matches[sampled_idx]]

        return sampled_proposals, sampled_labels, sampled_gt_boxes

    def forward(self, fpn_features: torch.Tensor, targets: Optional[torch.Tensor] = None):
        batch_size = fpn_features.shape[0]
        device = fpn_features.device
        image_size = self.image_size

        # -------- RPN forward --------
        anchors = self.anchor_generator(fpn_features, image_size)
        objectness, pred_bbox_deltas = self.rpn_head(fpn_features)

        objectness = objectness.permute(0, 2, 3, 1).reshape(batch_size, -1)
        pred_bbox_deltas = pred_bbox_deltas.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

        proposals = self.box_coder.decode(
            pred_bbox_deltas.reshape(-1, 4),
            anchors.repeat(batch_size, 1),
        ).reshape(batch_size, -1, 4)

        # -------- Training: return losses --------
        if self.training and targets is not None:
            losses: Dict[str, torch.Tensor] = {}

            total_rpn_cls_loss = 0.0
            total_rpn_box_loss = 0.0
            total_roi_cls_loss = 0.0
            total_roi_box_loss = 0.0

            for b in range(batch_size):
                # Normalized GT -> pixel coordinates
                gt_boxes = targets[b:b + 1] * torch.tensor(
                    [image_size[1], image_size[0], image_size[1], image_size[0]],
                    device=device,
                    dtype=torch.float32,
                )

                labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, gt_boxes)
                regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

                pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
                neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

                num_pos = int(self.rpn_batch_size_per_image * self.rpn_positive_fraction)
                num_pos = min(num_pos, pos_idx.numel())
                num_neg = self.rpn_batch_size_per_image - num_pos
                num_neg = min(num_neg, neg_idx.numel())

                if num_pos > 0:
                    perm_pos = torch.randperm(pos_idx.numel(), device=device)[:num_pos]
                    pos_idx_sampled = pos_idx[perm_pos]
                else:
                    pos_idx_sampled = pos_idx[:0]
                if num_neg > 0:
                    perm_neg = torch.randperm(neg_idx.numel(), device=device)[:num_neg]
                    neg_idx_sampled = neg_idx[perm_neg]
                else:
                    neg_idx_sampled = neg_idx[:0]

                sampled_idx = torch.cat([pos_idx_sampled, neg_idx_sampled])

                if sampled_idx.numel() > 0:
                    rpn_cls_loss = F.binary_cross_entropy_with_logits(
                        objectness[b][sampled_idx],
                        labels[sampled_idx],
                    )
                    total_rpn_cls_loss += rpn_cls_loss

                if pos_idx_sampled.numel() > 0:
                    rpn_box_loss = F.smooth_l1_loss(
                        pred_bbox_deltas[b][pos_idx_sampled],
                        regression_targets[pos_idx_sampled],
                        beta=1.0 / 9.0,
                    )
                    total_rpn_box_loss += rpn_box_loss

                filtered_proposals, _ = self.filter_proposals(
                    proposals[b:b + 1], objectness[b:b + 1], image_size
                )

                if len(filtered_proposals[0]) > 0:
                    sampled_proposals, sampled_labels, sampled_gt_boxes = self.sample_proposals(
                        filtered_proposals[0], targets[b], image_size
                    )

                    if sampled_proposals.numel() > 0:
                        class_logits, box_regression = self.roi_heads(
                            fpn_features[b:b + 1], sampled_proposals, image_size
                        )

                        roi_cls_loss = F.cross_entropy(class_logits, sampled_labels)
                        total_roi_cls_loss += roi_cls_loss

                        pos_mask = sampled_labels == 1
                        if pos_mask.any():
                            roi_regression_targets = self.box_coder.encode(
                                sampled_gt_boxes[pos_mask], sampled_proposals[pos_mask]
                            )
                            roi_box_loss = F.smooth_l1_loss(
                                box_regression[pos_mask],
                                roi_regression_targets,
                                beta=1.0 / 9.0,
                            )
                            total_roi_box_loss += roi_box_loss

            losses['rpn_objectness_loss'] = total_rpn_cls_loss / batch_size
            losses['rpn_box_loss'] = total_rpn_box_loss / batch_size
            losses['roi_cls_loss'] = total_roi_cls_loss / batch_size
            losses['roi_box_loss'] = total_roi_box_loss / batch_size
            losses['total_loss'] = (
                losses['rpn_objectness_loss']
                + losses['rpn_box_loss']
                + losses['roi_cls_loss']
                + losses['roi_box_loss']
            )
            return losses

        # -------- Inference: return best box per image --------
        filtered_proposals, scores = self.filter_proposals(proposals, objectness, image_size)

        results: List[Dict[str, torch.Tensor]] = []
        for b in range(batch_size):
            if len(filtered_proposals[b]) > 0:
                class_logits, box_regression = self.roi_heads(
                    fpn_features[b:b + 1], filtered_proposals[b], image_size
                )

                pred_scores = F.softmax(class_logits, dim=-1)[:, 1]
                pred_boxes = self.box_coder.decode(box_regression, filtered_proposals[b])
                pred_boxes = box_ops.clip_boxes_to_image(pred_boxes, image_size)

                keep = nms(pred_boxes, pred_scores, self.box_nms_thresh)
                keep = keep[:100]

                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                if len(pred_boxes) > 0:
                    best_idx = pred_scores.argmax()
                    best_box = pred_boxes[best_idx]
                    best_score = pred_scores[best_idx]

                    # Normalize to [0, 1] for downstream JSON export
                    best_box_norm = best_box / torch.tensor(
                        [image_size[1], image_size[0], image_size[1], image_size[0]],
                        device=device,
                        dtype=torch.float32,
                    )

                    results.append({'boxes': best_box_norm, 'scores': best_score})
                else:
                    results.append({'boxes': torch.zeros(4, device=device), 'scores': torch.tensor(0.0, device=device)})
            else:
                results.append({'boxes': torch.zeros(4, device=device), 'scores': torch.tensor(0.0, device=device)})

        return results