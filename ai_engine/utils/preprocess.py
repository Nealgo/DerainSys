import os
import glob
import argparse
from PIL import Image
from tqdm import tqdm
import shutil

def preprocess_dataset(source_rain, source_gt, dest_root, stitch=False):
    """
    Standardize dataset:
    1. Resize/Crop images to img_size.
    2. Rename to rain-x.png and norain-x.png.
    3. (Optional) Stitch rain and gt horizontally for visualization.
    """
    os.makedirs(os.path.join(dest_root, 'rain'), exist_ok=True)
    os.makedirs(os.path.join(dest_root, 'gt'), exist_ok=True)
    if stitch:
        os.makedirs(os.path.join(dest_root, 'preview_stitched'), exist_ok=True)

    # Get all images (assuming matching names in source folders, or simple sorting)
    # Supported extensions
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    rain_files = []
    for ext in exts:
        rain_files.extend(glob.glob(os.path.join(source_rain, ext)))
    
    # Sort to ensure alignment if using index
    rain_files = sorted(rain_files)
    
    print(f"Found {len(rain_files)} rain images.")

    for idx, rain_path in enumerate(tqdm(rain_files)):
        basename = os.path.basename(rain_path)
        # Try to find corresponding GT
        # Assumption: GT has the same filename as Rain
        gt_path = os.path.join(source_gt, basename)
        
        if not os.path.exists(gt_path):
            # Try replacing 'rain' with 'norain' if convention exists
            if 'rain' in basename:
                gt_basename = basename.replace('rain', 'norain')
                gt_path = os.path.join(source_gt, gt_basename)
            
            if not os.path.exists(gt_path):
                 # Fallback: try removing 'rain-' prefix? 
                 # Or assume listdir sorted matches (risky).
                 # For now, skip if strict match fails
                 print(f"Warning: GT not found for {basename}, skipping.")
                 continue

        # Process
        try:
            img_rain = Image.open(rain_path).convert('RGB')
            img_gt = Image.open(gt_path).convert('RGB')

            # Orientation Standardize: Rotate 90 degrees if portrait (H > W)
            # This ensures all images are "landscape" before resizing
            if img_rain.height > img_rain.width:
                img_rain = img_rain.transpose(Image.ROTATE_90)
                img_gt = img_gt.transpose(Image.ROTATE_90)

            # Resize removed to keep original size.
            # Logic: Only rotate if portrait.
            # If you check your dataset, make sure they are indeed pairs of (320x480) or (480x320)
            # After this rotation, they should all be 480x320 (Landscape)

            # Generate new names
            new_rain_name = f"{idx+1}.png"
            new_gt_name = f"{idx+1}.png"

            # Save
            img_rain.save(os.path.join(dest_root, 'rain', new_rain_name))
            img_gt.save(os.path.join(dest_root, 'gt', new_gt_name))

            # Stitch if requested ("Unify horizontal arrangement" might mean this?)
            if stitch:
                w, h = img_rain.size
                stitched_w = w * 2
                stitched_h = h
                stitched_img = Image.new('RGB', (stitched_w, stitched_h))
                stitched_img.paste(img_rain, (0, 0))
                stitched_img.paste(img_gt, (w, 0))
                stitched_img.save(os.path.join(dest_root, 'preview_stitched', f"pair-{idx+1:03d}.png"))

        except Exception as e:
            print(f"Error processing {basename}: {e}")

    print("Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Derain Dataset")
    parser.add_argument("--src_rain", type=str, required=True, help="Source Rain images folder")
    parser.add_argument("--src_gt", type=str, required=True, help="Source GT images folder")
    parser.add_argument("--dest", type=str, default="dataset/processed_train", help="Destination folder")
    parser.add_argument("--size", type=int, default=256, help="Target image size (square)")
    parser.add_argument("--stitch", action='store_true', help="Generate side-by-side preview images (Left: Rain, Right: GT)")
    
    args = parser.parse_args()
    
    preprocess_dataset(args.src_rain, args.src_gt, args.dest, args.stitch)
