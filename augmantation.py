import os
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from tqdm import tqdm

# ====== CONFIGURATION ======
INPUT_FOLDER = r"C:\Users\User\Downloads\Breast\benign"  # Change per run
OUTPUT_FOLDER = r"C:\Users\User\Downloads\Breast\normal-benign"  # Change per run
TARGET_COUNT = 2000  # Number of images you want in output

# ====== AUGMENTATION FUNCTIONS ======
def random_flip(image):
    if random.choice([True, False]):
        return ImageOps.mirror(image)
    return image

def random_rotation(image):
    return image.rotate(random.randint(-20, 20))

def random_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(0.7, 1.3))

def random_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(random.uniform(0.7, 1.3))

def random_crop(image):
    width, height = image.size
    crop_percent = random.uniform(0.85, 1.0)
    new_width = int(width * crop_percent)
    new_height = int(height * crop_percent)
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    return image.crop((left, top, left + new_width, top + new_height)).resize((width, height))

def augment_image(image):
    image = random_flip(image)
    image = random_rotation(image)
    image = random_brightness(image)
    image = random_contrast(image)
    image = random_crop(image)
    return image

# ====== MAIN FUNCTION ======
def process_folder(input_folder, output_folder, target_count):
    os.makedirs(output_folder, exist_ok=True)

    # Get all image paths
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"  No images found in {input_folder}")
        return
    
    print(f"=Â Found {len(image_files)} images in '{input_folder}'")
    
    # Copy original images first
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)
        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(output_folder, img_file))

    # Augment until target count is reached
    counter = len(image_files)
    with tqdm(total=target_count, initial=counter, desc="Processing") as pbar:
        while counter < target_count:
            img_file = random.choice(image_files)
            img_path = os.path.join(input_folder, img_file)
            img = Image.open(img_path).convert("RGB")

            aug_img = augment_image(img)
            aug_name = f"aug_{counter}.jpg"
            aug_img.save(os.path.join(output_folder, aug_name))

            counter += 1
            pbar.update(1)

    print(f" Balanced dataset saved at: {output_folder}")

# ====== RUN SCRIPT ======
if __name__ == "__main__":
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_COUNT)
