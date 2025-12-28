import sys
import torch
from PIL import Image
from torchvision import transforms
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ai_engine'))
from Derain_model import UNet

def restore(input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3).to(device)
    model_path = os.path.join(os.path.dirname(__file__), '../ai_engine/pth/Derain_model.pth')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = Image.open(input_path).convert('RGB')
    w, h = img.size
    
    # Use a Safe Margin to push boundary artifacts out of the valid image area
    # Even if dimensions are aligned, we force padding to handle model boundary effects
    align = 16
    safe_margin = 16 # Add 16 pixels safety zone on all sides
    
    # Calculate target dimensions that are divisible by 16 AND include the margin
    target_w = ((w + 2 * safe_margin + align - 1) // align) * align
    target_h = ((h + 2 * safe_margin + align - 1) // align) * align
    
    pad_total_w = target_w - w
    pad_total_h = target_h - h
    
    # Center the image in the padded tensor
    pad_left = pad_total_w // 2
    pad_right = pad_total_w - pad_left
    pad_top = pad_total_h // 2
    pad_bottom = pad_total_h - pad_top
    
    import torch.nn.functional as F
    
    # Convert image to Tensor first
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # (left, right, top, bottom)
    input_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

    with torch.no_grad():
        output = model(input_tensor)
        
    # Crop strictly to the original w, h, discarding all padding (including the safe margin)
    output = output[:, :, pad_top:pad_top+h, pad_left:pad_left+w]
        
    output_img = transforms.ToPILImage()(output.squeeze(0).clamp(0, 1))
    output_img.save(output_path)

if __name__ == '__main__':
    restore(sys.argv[1], sys.argv[2]) 