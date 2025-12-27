import sys
import torch
from PIL import Image
from torchvision import transforms
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_engine'))
from mamba_model import UNet

def restore(input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    # Model is now expected to be in ../ai_engine/pth/
    model_path = os.path.join(os.path.dirname(__file__), '../ai_engine/pth/unet_mamba_best_psnr.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(input_path).convert('RGB')
    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
    output_img = transforms.ToPILImage()(output.squeeze(0).clamp(0, 1))
    output_img = output_img.resize(original_size, Image.BICUBIC)
    output_img.save(output_path)

if __name__ == '__main__':
    restore(sys.argv[1], sys.argv[2]) 