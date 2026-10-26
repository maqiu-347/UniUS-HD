import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import glob
import cv2
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.encoders import get_encoder

# ================== 配置参数（与原始 MAE 对齐） ==================
BATCH_SIZE = 20               
NUM_EPOCHS = 30
DATA_ROOT = '/home/wujinjing/wjj/jj/us_dataset/train'  
SAVE_PATH = '/home/wujinjing/wjj/jj/_mae_new/mae_tradition_mit.pth'                   

MASK_RATIO = 0.75
IMG_SIZE = 512
PATCH_SIZE = 32                        # MiT 输出 16x16 网格，patch 大小设为 32
EMBED_DIM = 512                        # 解码器维度（与原始 MAE 解码器一致）
DECODER_EMBED_DIM = 512
DECODER_DEPTH = 8
DECODER_NUM_HEADS = 16
NORM_PIX_LOSS = True

LR_BASE = 1e-4
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)
WARMUP_EPOCHS = 5

# ================== 辅助函数：sincos 位置编码 ==================
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack([grid_h, grid_w], dim=0).reshape([2, -1])
    pos = grid.permute(1, 0)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, pos[:, 0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, pos[:, 1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1)
    if cls_token:
        cls_pos_embed = torch.zeros(1, embed_dim)
        pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.)
    omega = 1. / (10000 ** omega)
    out = torch.einsum('i,j->ij', pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

# ================== 基础模块（与原始 MAE 一致） ==================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ================== 对齐的 MAE 模型（MiT-B5 编码器） ==================
class MAE_MiT(nn.Module):
    def __init__(self):
        super().__init__()
        # ---------- 编码器：MiT-B5（利用 ImageNet 预训练初始化） ----------
        self.encoder = get_encoder(
            name='mit_b5',
            weights='imagenet',
            in_channels=3,
            depth=5,                      # 输出 5 个特征（0:stem, 1~4:stages）
        )
        encoder_out_channels = self.encoder.out_channels[-1]  # 512
        # 将 MiT 输出映射到解码器维度
        self.encoder_to_decoder = nn.Linear(encoder_out_channels, DECODER_EMBED_DIM, bias=True)

        # ---------- 解码器（与原始 MAE 完全一致） ----------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DECODER_EMBED_DIM))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DECODER_EMBED_DIM))

        # 位置编码：网格尺寸 = IMG_SIZE // PATCH_SIZE = 16
        self.grid_size = IMG_SIZE // PATCH_SIZE  # 16
        decoder_pos_embed = get_2d_sincos_pos_embed(DECODER_EMBED_DIM, self.grid_size, cls_token=True)
        self.register_buffer('decoder_pos_embed', decoder_pos_embed.unsqueeze(0))

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(DECODER_EMBED_DIM, DECODER_NUM_HEADS, mlp_ratio=4, qkv_bias=True)
            for _ in range(DECODER_DEPTH)
        ])
        self.decoder_norm = nn.LayerNorm(DECODER_EMBED_DIM, eps=1e-6)
        self.decoder_pred = nn.Linear(DECODER_EMBED_DIM, PATCH_SIZE * PATCH_SIZE * 3, bias=True)

        self.norm_pix_loss = NORM_PIX_LOSS
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """将图像转为 patch 序列 [B, N, p*p*C]"""
        p = PATCH_SIZE
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p * p * 3)

    def unpatchify(self, x):
        p = PATCH_SIZE
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], 3, h * p, w * p)

    def random_masking(self, x, mask_ratio):
        """随机遮蔽（丢弃token），与原始MAE完全一致"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # 1. 对输入图像做像素级遮蔽（直接置零）
        #    为保持与原始 MAE token 丢弃的一致性，我们在编码器前先遮蔽图像
        #    但 MiT 编码器对完整分辨率图像操作，需要将对应 patch 区域的像素置零
        #    这里采用简单方式：生成 mask 后遮蔽图像，再输入编码器
        B, C, H, W = x.shape
        patch_h = self.grid_size
        patch_w = self.grid_size
        num_patches = patch_h * patch_w
        len_keep = int(num_patches * (1 - mask_ratio))

        # 随机生成遮蔽的 patch 索引
        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        # 构建像素级 mask
        mask_pixel = torch.ones(B, 1, H, W, device=x.device)
        p = PATCH_SIZE
        for b in range(B):
            for idx in ids_keep[b]:
                hi = idx // patch_w
                wi = idx % patch_w
                mask_pixel[b, 0, hi*p:(hi+1)*p, wi*p:(wi+1)*p] = 0.0  # 保留的区域置0（不遮蔽）

        # 遮蔽图像（保留区域保持原值，遮蔽区域置0）
        x_masked = x * (1 - mask_pixel)

        # 2. 送入 MiT 编码器
        features = self.encoder(x_masked)                 # 列表 [stem, stage1, stage2, stage3, stage4]
        feat = features[-1]                               # 取最后一层 stage4: [B, 512, H/32, W/32]
        # feat 尺寸应为 [B, 512, 16, 16] (H=512, stride=32)
        # 转成 token 序列
        B, D, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)          # [B, 256, 512]

        # 投影到解码器维度
        tokens = self.encoder_to_decoder(tokens)          # [B, 256, decoder_dim]

        # 重新进行随机遮蔽（token 级），与原始 MAE 一致
        x_masked_tokens, mask, ids_restore = self.random_masking(tokens, mask_ratio)

        # 添加 cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x_masked_tokens), dim=1)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # x: [B, 1+len_keep, decoder_dim]
        # 补全 mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_cls = x[:, :1, :]
        x_patches = x[:, 1:, :]
        x_ = torch.cat([x_patches, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x_cls, x_], dim=1) + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]  # 去掉 cls，只保留 patch 预测

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)          # [B, 256, 32*32*3]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

# ================== 数据集（与原始 MAE 一致） ==================
class ImageOnlyDataset(Dataset):
    def __init__(self, data_root, transforms=None):
        self.transforms = transforms
        self.csv_path = os.path.join(data_root, 'csv_files')
        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, record['image_path']))
        image = cv2.imread(image_abs_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        return {'image': image}

# ================== 训练函数（与原始 MAE 完全对齐） ==================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("MAE Pre-training with MiT-B5 (Aligned with Official MAE)")
    print(f"Device: {device}")
    print(f"Mask Ratio: {MASK_RATIO}, Patch Size: {PATCH_SIZE}")
    print(f"Norm Pix Loss: {NORM_PIX_LOSS}")
    print(f"{'='*60}\n")

    # 数据增强（与官方 MAE 一致）
    transforms = A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.2, 1.0), interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset = ImageOnlyDataset(DATA_ROOT, transforms=transforms)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=lambda batch: {'image': torch.stack([b['image'] for b in batch])}
    )

    print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}\n")

    # 模型
    model = MAE_MiT().to(device)

    # 优化器（与官方 MAE 一致）
    eff_batch_size = BATCH_SIZE
    lr = LR_BASE * eff_batch_size / 256.0
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)

    # 学习率调度（warmup + cosine）
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
            return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in loop:
            images = batch['image'].to(device)
            loss, pred, mask = model(images, mask_ratio=MASK_RATIO)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} Summary: Total Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'encoder': model.encoder.state_dict(),   # 只保存编码器（供下游使用）
                'model_state_dict': model.state_dict(),  # 保存整个模型（可选）
                'epoch': epoch,
                'loss': best_loss,
            }, SAVE_PATH)
            print(f"✅ Saved! Best Loss: {best_loss:.4f}\n")

        scheduler.step()

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Encoder saved to: {SAVE_PATH}")
    print("="*60)

if __name__ == '__main__':
    train()