import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

# 导入修改后的模型和指标
from Derain_model import UNet 
from metrics import calculate_psnr, calculate_ssim 

# ==========================================
# 1. Edge Loss (边缘损失) - 提分神器
# ==========================================
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .05], [.25, -1.2, .25], [.05, .25, .05]])
        self.kernel = k.unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        x_grad = F.conv2d(x, self.kernel, padding=1, groups=3)
        y_grad = F.conv2d(y, self.kernel, padding=1, groups=3)
        return self.loss(x_grad, y_grad)

# 数据集代码 (保持你现在的 RandomCrop 逻辑即可，这里略去以节省篇幅)
# 请直接复制之前有效的 RainDataset 代码
class RainDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, is_train=True):
        self.rain_dir = os.path.join(root_dir, 'rain')
        self.gt_dir = os.path.join(root_dir, 'gt')
        self.rain_names = sorted([f for f in os.listdir(self.rain_dir) if f.lower().endswith(('.png', '.jpg'))])
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.rain_names)

    def __getitem__(self, idx):
        rain_name = self.rain_names[idx]
        gt_name = rain_name 

        rain_path = os.path.join(self.rain_dir, rain_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        rain_img = Image.open(rain_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')

        if self.is_train:
            w, h = rain_img.size
            th, tw = self.patch_size, self.patch_size
            if w < tw or h < th:
                rain_img = transforms.Resize((max(h, th), max(w, tw)))(rain_img)
                gt_img = transforms.Resize((max(h, th), max(w, tw)))(gt_img)
                w, h = rain_img.size
            
            i = torch.randint(0, h - th + 1, (1,)).item()
            j = torch.randint(0, w - tw + 1, (1,)).item()
            
            rain_img = transforms.functional.crop(rain_img, i, j, th, tw)
            gt_img = transforms.functional.crop(gt_img, i, j, th, tw)
            
            # 数据增强：随机翻转
            if torch.rand(1) < 0.5:
                rain_img = transforms.functional.hflip(rain_img)
                gt_img = transforms.functional.hflip(gt_img)
            if torch.rand(1) < 0.5:
                rain_img = transforms.functional.vflip(rain_img)
                gt_img = transforms.functional.vflip(gt_img)

        t_rain = transforms.ToTensor()(rain_img)
        t_gt = transforms.ToTensor()(gt_img)
        return t_rain, t_gt

# Charbonnier Loss
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

if __name__ == "__main__":
    # === 参数优化 ===
    # 增加 Epoch，因为使用了 Plateau 调度器，需要更久时间来收敛
    EPOCHS = 150 
    BATCH_SIZE = 16 
    LR = 2e-4
    # 如果显存允许，尝试把 Patch Size 改为 256 (会显著提升效果)
    PATCH_SIZE = 128 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = UNet().to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # === 修改策略：使用 Plateau 调度器 ===
    # 当 PSNR 不再上升时，降低学习率。这比 Cosine 更稳。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss().to(device)

    # 数据集路径
    DATASET_DIR = 'dataset/train' 
    train_ds = RainDataset(DATASET_DIR, patch_size=PATCH_SIZE, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    best_psnr = 0
    save_dir = "results_vis_improved"
    os.makedirs(save_dir, exist_ok=True)

    print("Starting Improved Training...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, (rain, gt) in enumerate(loop):
            rain, gt = rain.to(device), gt.to(device)
            
            pred = model(rain)
            
            # === 混合 Loss ===
            # 0.05 的边缘 Loss 权重足以强迫网络关注高频细节
            loss_pixel = criterion_char(pred, gt)
            loss_edge = criterion_edge(pred, gt)
            loss = loss_pixel + 0.05 * loss_edge
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 监控 PSNR
            with torch.no_grad():
                # 简单计算一个batch的均值用于显示
                # 实际上应该在验证集上计算，这里用训练集近似监控
                p = calculate_psnr(pred.clamp(0,1).cpu(), gt.clamp(0,1).cpu())
                epoch_psnr += p

            loop.set_postfix(loss=loss.item(), psnr=p)

        avg_loss = epoch_loss / len(train_loader)
        avg_psnr = epoch_psnr / len(train_loader)
        
        # === 更新学习率 ===
        # 注意：这里我们监控 PSNR，如果 PSNR 不涨了，就降低 LR
        scheduler.step(avg_psnr)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.5f}, Avg PSNR={avg_psnr:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), "best_unet_cbam.pth")
            print(f"🔥 New Best PSNR: {best_psnr:.4f}")
            
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                vis = torch.cat([rain[0:2], pred[0:2].clamp(0,1), gt[0:2]], dim=0)
                vutils.save_image(vis, f"{save_dir}/epoch_{epoch+1}.png", nrow=2)