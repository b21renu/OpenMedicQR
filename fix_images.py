# fix_images.py
import os
from PIL import Image

def clean_images(directory):
    """
    Opens and re-saves all JPG images in a directory to fix corruption issues.
    """
    print(f"--- Starting image cleaning for directory: {directory} ---")
    cleaned_count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            filepath = os.path.join(directory, filename)
            try:
                # Open the image using Pillow
                with Image.open(filepath) as img:
                    # Re-save the image. This process cleans the file structure.
                    img.save(filepath, 'JPEG')
                cleaned_count += 1
            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")
    
    print(f"Cleaned {cleaned_count} images in {directory}.\n")

# --- Main execution ---
# Add all your image folders to this list
image_directories = [
    'data/demo_images/train_images',
    'data/demo_images/test_images'
]

for img_dir in image_directories:
    if os.path.exists(img_dir):
        clean_images(img_dir)
    else:
        print(f"Warning: Directory not found, skipping: {img_dir}")
        
print("--- Image cleaning process finished. ---")