import os
import argparse
import shutil
import random
from glob import glob
from math import ceil

def split_dataset_with_labels(emotion_dirs, train_output, val_output, test_output, test_size=0.2, val_size=0.5, seed=42):
    """
    Split images from emotion folders into train/val/test with class subfolders.

    Parameters:
        emotion_dirs (list): List of paths to emotion folders.
        train_output (str): Output folder for training images.
        val_output (str): Output folder for validation images.
        test_output (str): Output folder for test images.
        test_size (float): Fraction of all images to reserve for val+test.
        val_size (float): Fraction of val+test to use for validation.
        seed (int): Random seed.
    """
    random.seed(seed)

    output_dirs = {"train": train_output, "val": val_output, "test": test_output}

    # Prepare output directories
    for split, out_dir in output_dirs.items():
        for folder in emotion_dirs:
            label = os.path.basename(folder)
            os.makedirs(os.path.join(out_dir, label), exist_ok=True)

    # Process each emotion folder
    for folder in emotion_dirs:
        label = os.path.basename(folder)
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend(glob(os.path.join(folder, ext)))
        random.shuffle(images)

        n_total = len(images)
        n_test_val = ceil(n_total * test_size)
        n_val = ceil(n_test_val * val_size)
        n_test = n_test_val - n_val
        n_train = n_total - n_test_val

        split_images = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        # Copy images to appropriate split folders
        for split_name, split_list in split_images.items():
            out_dir = output_dirs[split_name]
            for img in split_list:
                shutil.copy(img, os.path.join(out_dir, label, os.path.basename(img)))

        print(f"[{label}] Split complete: train={len(split_images['train'])}, val={len(split_images['val'])}, test={len(split_images['test'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', required=True, help="Paths to emotion folders")
    parser.add_argument("--train_output", type=str, required=True, help="Output folder for training images")
    parser.add_argument("--val_output", type=str, required=True, help="Output folder for validation images")
    parser.add_argument("--test_output", type=str, required=True, help="Output folder for test images")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of total data reserved for val+test")
    parser.add_argument("--val_size", type=float, default=0.5, help="Fraction of val+test reserved for validation")
    args = parser.parse_args()

    split_dataset_with_labels(
        emotion_dirs=args.datasets,
        train_output=args.train_output,
        val_output=args.val_output,
        test_output=args.test_output,
        test_size=args.test_size,
        val_size=args.val_size
    )