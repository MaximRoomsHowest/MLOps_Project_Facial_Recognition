import os
import argparse
import shutil
import random
from glob import glob
from math import ceil

def split_combined_dataset(emotion_dirs, train_output, val_output, test_output, test_size=0.2, val_size=0.5, seed=42):
    """
    Combine images from multiple emotion folders, shuffle, and split into train, val, and test.

    Parameters:
        emotion_dirs (list): List of 5 emotion folder paths.
        train_output (str): Output folder for training images.
        val_output (str): Output folder for validation images.
        test_output (str): Output folder for test images.
        test_size (float): Fraction of all images to reserve for val+test.
        val_size (float): Fraction of val+test to use for validation.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    # Gather all images from all emotion folders
    all_images = []
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    for folder in emotion_dirs:
        for ext in image_extensions:
            all_images.extend(glob(os.path.join(folder, ext)))

    print(f"Total images found: {len(all_images)}")
    random.shuffle(all_images)


    # Compute split sizes
    n_total = len(all_images)
    n_test_val = ceil(n_total * test_size)
    n_val = ceil(n_test_val * val_size)
    n_test = n_test_val - n_val
    n_train = n_total - n_test_val

    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train+n_val:]

    # Create output folders
    for out_dir in [train_output, val_output, test_output]:
        os.makedirs(out_dir, exist_ok=True)

    # Copy images
    for img in train_images:
        shutil.copy(img, os.path.join(train_output, os.path.basename(img)))
    for img in val_images:
        shutil.copy(img, os.path.join(val_output, os.path.basename(img)))
    for img in test_images:
        shutil.copy(img, os.path.join(test_output, os.path.basename(img)))

    print(f"Split complete:")
    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs=5, required=True, help="Paths to 5 emotion folders")
    parser.add_argument("--train_output", type=str, required=True, help="Output folder for training images")
    parser.add_argument("--val_output", type=str, required=True, help="Output folder for validation images")
    parser.add_argument("--test_output", type=str, required=True, help="Output folder for test images")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of total data reserved for val+test")
    parser.add_argument("--val_size", type=float, default=0.5, help="Fraction of val+test reserved for validation")
    args = parser.parse_args()

    split_combined_dataset(
        emotion_dirs=args.datasets,
        train_output=args.train_output,
        val_output=args.val_output,
        test_output=args.test_output,
        test_size=args.test_size,
        val_size=args.val_size
    )

if __name__ == "__main__":
    main()