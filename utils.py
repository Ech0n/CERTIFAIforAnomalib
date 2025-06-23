import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_masks(image_dataset, mask_root_dir,transform):
    masks = []
    for path, _ in image_dataset.imgs:
        filename = os.path.basename(path)
        class_folder = os.path.basename(os.path.dirname(path))
        filename = filename.replace(".","_mask.")
        if class_folder != "good":
            mask_path = os.path.join(mask_root_dir, class_folder, filename)

            mask = Image.open(mask_path)
            mask_tensor = transform(mask)  # Shape: [1, 300, 300]
            masks.append(mask_tensor.squeeze(0).numpy())  # Shape: [300, 300]
    return np.stack(masks)

def get_fixed_indices_from_mask(mask):
    fixed = []
    n , h, w = mask.shape
    for row in range(h):
        for col in range(w):
            if mask[0,row, col] == 0:  # Background pixel
                # Add index for each RGB channel
                for c in range(3):
                    flat_index = c * h * w + row * w + col
                    fixed.append(flat_index)
    return fixed

def save_image(image_array, filename):
    if image_array.shape[0] != 3:
        raise ValueError("Input must have shape (3, H, W)")

    # Convert from (3, H, W) to (H, W, 3)
    img = np.transpose(image_array, (1, 2, 0))

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

    Image.fromarray(img).save(filename)

def load_images_from_directory(base_dir, image_size=(1024, 1024)):
    image_data = []
    labels = []

    for label_name in os.listdir(base_dir):
        label_path = os.path.join(base_dir, label_name)
        if os.path.isdir(label_path):
            for img_file in tqdm(os.listdir(label_path), desc=f"Loading {label_name}"):
                img_path = os.path.join(label_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        img = img.resize(image_size)
                        img_array = np.array(img)
                        image_data.append(img_array)
                        labels.append(label_name)
                except Exception as e:
                    print(f"Failed to load image {img_path}: {e}")

    return np.array(image_data), np.array(labels)

def extract_numpy_from_loader(dataloader):
    all_images = []
    all_labels = []
    for images, labels in dataloader:
        all_images.append(images.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_images), np.concatenate(all_labels)
