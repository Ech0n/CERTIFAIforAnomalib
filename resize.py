import os
from PIL import Image

root_dir = '/mnt/f/datasets/mvtec/bottle_small'
max_size = (256, 256)

def resize_image(image_path, max_size):
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(image_path)
            print(f"Resized: {image_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Traverse directories
for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            full_path = os.path.join(dirpath, file)
            resize_image(full_path, max_size)
