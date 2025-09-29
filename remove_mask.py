import os
import shutil

# Original dataset path
src_dataset_path = r"C:\Users\User\Downloads\archive (7)\Dataset_BUSI_with_GT\benign"

# New dataset path without masks
dst_dataset_path = r"C:\Users\User\Downloads\Breast\benign"

# Create destination folder if not exists
os.makedirs(dst_dataset_path, exist_ok=True)

# Walk through all subfolders
for root, dirs, files in os.walk(src_dataset_path):
    # Get relative path from original dataset
    relative_path = os.path.relpath(root, src_dataset_path)
    
    # Create the same folder structure in the destination
    dst_folder = os.path.join(dst_dataset_path, relative_path)
    os.makedirs(dst_folder, exist_ok=True)
    
    for file in files:
        # Skip files containing '_mask' in the filename
        if "_mask" not in file.lower():
            src_file_path = os.path.join(root, file)
            dst_file_path = os.path.join(dst_folder, file)
            shutil.copy2(src_file_path, dst_file_path)

print(f" New dataset created at: {dst_dataset_path} (without mask images)")
