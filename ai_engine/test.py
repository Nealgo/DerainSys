import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# UNet model definition
from mamba_model import UNet # Import the UNet model

# Import PSNR and SSIM calculation functions.
# Assuming 'utils.py' is in the same directory and contains these functions.
# You might need to provide the content of utils.py if it's not standard.
from metrics import calculate_psnr, calculate_ssim 
import math # Only needed if gaussian is used elsewhere, not directly in this script.

# The gaussian function is not directly used in this test script's main logic,
# but keeping it here for completeness if it's part of your project's utilities.
def gaussian(window_size, sigma):
    # This function seems unrelated to the current test script's execution,
    # but I'll keep it as it was in your provided code.
    gauss = torch.tensor([
        math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)
    ])
    return gauss/gauss.sum()

class RainDataset(Dataset):
    def __init__(self, root_dir):
        self.rain_dir = os.path.join(root_dir, 'rain')
        self.gt_dir = os.path.join(root_dir, 'gt')
        
        # Collect all 'rain-xxx.png' files in the rain directory.
        self.rain_names = sorted([f for f in os.listdir(self.rain_dir) if f.startswith('rain-')])
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # Resize to 256x256
            transforms.ToTensor()          # Convert to PyTorch Tensor
        ])

    def __len__(self):
        return len(self.rain_names)

    def __getitem__(self, idx):
        rain_name = self.rain_names[idx] # This is the filename you want to return
        
        # Extract the numeric part (e.g., '001', '002') from 'rain-xxx.png'
        num_str = rain_name.replace('rain-', '').replace('.png', '')
        # Construct the corresponding ground truth filename 'norain-xxx.png'
        gt_name = f'norain-{num_str}.png'
        
        rain_img_path = os.path.join(self.rain_dir, rain_name)
        gt_img_path = os.path.join(self.gt_dir, gt_name)

        # Open and convert images to RGB
        rain_img = Image.open(rain_img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('RGB')
        
        # Apply transformations and return the image tensors and the original filename
        return rain_name, self.transform(rain_img), self.transform(gt_img) # Changed 'name' to 'rain_name'

# --- Main Test Script ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device for testing: {device}")

# Define your dataset root directory for testing
# IMPORTANT: Replace 'dataset/test' with the actual path to your test dataset
TEST_DATASET_ROOT_DIR = 'dataset/test' 

if not os.path.exists(TEST_DATASET_ROOT_DIR):
    print(f"Error: Test dataset directory '{TEST_DATASET_ROOT_DIR}' not found.")
    print("Please make sure your test dataset structure is like:")
    print("  your_project_folder/")
    print("  └── dataset/")
    print("      └── test/")
    print("          ├── rain/ (contains rain-X.png)")
    print("          └── gt/   (contains norain-X.png)")
    exit() # Exit if dataset not found

test_dataset = RainDataset(TEST_DATASET_ROOT_DIR)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size is 1 for testing

# Initialize the UNet model
# Ensure in_channels and out_channels match your trained model
model = UNet(in_channels=3, out_channels=3).to(device) 

# Define the path to your trained model weights
model_path = "unet_mamba_final.pth" # Use the name you saved your trained model as

if not os.path.exists(model_path):
    print(f"Error: Model weights file '{model_path}' not found.")
    print("Please ensure you have trained your model and saved its weights to this path.")
    exit() # Exit if model weights not found

# Load the trained model weights
print(f"Loading model weights from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # Set the model to evaluation mode (disables dropout, BatchNorm updates, etc.)

# PSNR and SSIM lists to store metrics for all test images
psnr_list, ssim_list = [], []

# Create a directory to save the denoised results
results_dir = 'results_unet_mamba' # Changed directory name to reflect the model
os.makedirs(results_dir, exist_ok=True)
print(f"Saving denoised images to: {results_dir}")

print("Starting evaluation...")
# Disable gradient calculations during inference for speed and memory efficiency
with torch.no_grad():
    for name, rainy_img, gt_img in test_loader:
        # Move tensors to the specified device (GPU if available)
        rainy_img, gt_img = rainy_img.to(device), gt_img.to(device)
        
        # Perform the forward pass. The UNet directly takes the rainy image.
        output = model(rainy_img)
        
        # Clamp output to [0, 1] range and move to CPU for metric calculation/saving
        output_cpu = output.squeeze(0).clamp(0, 1).cpu()
        gt_cpu = gt_img.squeeze(0).cpu() # Also move gt to CPU for metric calculation

        # Calculate PSNR and SSIM
        psnr = calculate_psnr(output_cpu, gt_cpu)
        ssim = calculate_ssim(output_cpu, gt_cpu)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        
        # Print metrics for the current image
        print(f"Processing {name[0]}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

        # Convert the output tensor to a PIL Image and save it as PNG
        out_img_pil = transforms.ToPILImage()(output_cpu)
        # Use the original filename (without path or extension) for saving
        # os.path.splitext(name[0])[0] gets 'rain-XXX' from 'rain-XXX.png'
        out_img_pil.save(os.path.join(results_dir, f'{os.path.splitext(name[0])[0]}.png'))

# Calculate and print average metrics
if psnr_list and ssim_list:
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    print(f"\n--- Evaluation Summary ---")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
else:
    print("\nNo images processed. Please check your dataset path and file names.")