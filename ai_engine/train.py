import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# Assuming your new UNet with Mamba is in unet_mamba_model.py
from mamba_model import UNet # Import the new UNet model
from metrics import calculate_psnr, calculate_ssim
import torchvision.utils as vutils
# import matplotlib.pyplot as plt  # 已不需要
from tqdm import tqdm

# 数据集定义
class RainDataset(Dataset):
    def __init__(self, root_dir):
        self.rain_dir = os.path.join(root_dir, 'rain')
        self.gt_dir = os.path.join(root_dir, 'gt')
        # Ensure that the sorting logic correctly pairs rain images with gt images
        # based on your file naming convention.
        # Assuming 'rain-X.png' corresponds to 'norain-X.png'
        self.rain_names = sorted([f for f in os.listdir(self.rain_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.rain_names)

    def __getitem__(self, idx):
        rain_name = self.rain_names[idx]
        # Direct mapping: assume GT has the exact same filename
        gt_name = rain_name

        rain_img_path = os.path.join(self.rain_dir, rain_name)
        gt_img_path = os.path.join(self.gt_dir, gt_name)

        rain_img = Image.open(rain_img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('RGB')
        
        return self.transform(rain_img), self.transform(gt_img)

# Charbonier Loss实现
class CharbonierLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.epsilon ** 2))

# 训练参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define your dataset root directory
# IMPORTANT: Replace 'dataset/train' with the actual path to your dataset
# where 'dataset/train/rain' and 'dataset/train/gt' exist.
DATASET_ROOT_DIR = 'dataset/train' 

if not os.path.exists(DATASET_ROOT_DIR):
    print(f"Error: Dataset directory '{DATASET_ROOT_DIR}' not found.")
    print("Please make sure your dataset structure is like:")
    print("  your_project_folder/")
    print("  └── dataset/")
    print("      └── train/")
    print("          ├── rain/ (contains rain-X.png)")
    print("          └── gt/   (contains norain-X.png)")
    exit() # Exit if dataset not found

train_dataset = RainDataset(DATASET_ROOT_DIR)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize the new UNet model
# in_channels=3 for RGB images, out_channels=3 for RGB output
model = UNet(in_channels=3, out_channels=3).to(device) 

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = CharbonierLoss()

num_epochs = 50

print("Starting training...")
best_psnr = 0
best_ssim = 0
best_psnr_epoch = 0
best_ssim_epoch = 0
# 用于保存最佳模型权重
best_psnr_path = ''
best_ssim_path = ''
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    vis_dir = f"results_vis/epoch_{epoch+1}"
    os.makedirs(vis_dir, exist_ok=True)
    train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (rainy_img, gt_img) in train_iter:
        rainy_img, gt_img = rainy_img.to(device), gt_img.to(device)
        output = model(rainy_img)
        loss = criterion(output, gt_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 计算PSNR和SSIM（取batch均值）
        with torch.no_grad():
            batch_psnr = 0
            batch_ssim = 0
            for i in range(rainy_img.size(0)):
                pred = output[i].clamp(0, 1).unsqueeze(0)
                gt = gt_img[i].clamp(0, 1).unsqueeze(0)
                batch_psnr += calculate_psnr(pred, gt)
                batch_ssim += calculate_ssim(pred, gt)
            batch_psnr /= rainy_img.size(0)
            batch_ssim /= rainy_img.size(0)
            total_psnr += batch_psnr
            total_ssim += batch_ssim

        # 可视化保存前2个batch
        if batch_idx < 2:
            for i in range(min(2, rainy_img.size(0))):
                vutils.save_image(rainy_img[i], f"{vis_dir}/input_{batch_idx}_{i}.png")
                vutils.save_image(output[i].clamp(0,1), f"{vis_dir}/output_{batch_idx}_{i}.png")
                vutils.save_image(gt_img[i], f"{vis_dir}/gt_{batch_idx}_{i}.png")

        if (batch_idx + 1) % 10 == 0:
            train_iter.set_postfix(loss=loss.item())

    avg_epoch_loss = total_loss / len(train_loader)
    avg_epoch_psnr = total_psnr / len(train_loader)
    avg_epoch_ssim = total_ssim / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} finished, Average Loss: {avg_epoch_loss:.4f}, PSNR: {avg_epoch_psnr:.4f}, SSIM: {avg_epoch_ssim:.4f}")

    # 更新前50个epoch的最高PSNR和SSIM，并保存最佳模型
    if avg_epoch_psnr > best_psnr:
        best_psnr = avg_epoch_psnr
        best_psnr_epoch = epoch + 1
        best_psnr_path = f"unet_mamba_best_psnr.pth"
        torch.save(model.state_dict(), best_psnr_path)
        print(f"[Best PSNR] Model saved to {best_psnr_path} at epoch {best_psnr_epoch}")
    if avg_epoch_ssim > best_ssim:
        best_ssim = avg_epoch_ssim
        best_ssim_epoch = epoch + 1
        best_ssim_path = f"unet_mamba_best_ssim.pth"
        torch.save(model.state_dict(), best_ssim_path)
        print(f"[Best SSIM] Model saved to {best_ssim_path} at epoch {best_ssim_epoch}")

    # Optional: Save model checkpoints
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"unet_mamba_epoch{epoch+1}.pth")
        print(f"Model saved to unet_mamba_epoch{epoch+1}.pth")

# 训练结束后保存最终模型
torch.save(model.state_dict(), "unet_mamba_final.pth")
print("\nTraining finished. Final model saved to unet_mamba_final.pth")

# 输出前50个epoch的最高PSNR和SSIM
print(f"前50个epoch最高PSNR: {best_psnr:.4f} (epoch {best_psnr_epoch})")
print(f"前50个epoch最高SSIM: {best_ssim:.4f} (epoch {best_ssim_epoch})")
if best_psnr_path:
    print(f"PSNR最优模型权重已保存: {best_psnr_path}")
if best_ssim_path:
    print(f"SSIM最优模型权重已保存: {best_ssim_path}")