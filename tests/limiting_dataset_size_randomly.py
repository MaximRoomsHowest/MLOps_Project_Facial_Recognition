import shutil
import numpy as np
import pandas as pd
from pathlib import Path

def create_df_by_classes(base_path, classes, img_limit=1000):
    dd = {"images": [], "labels": []}
    base_path = Path(base_path)

    for class_name in classes:
        img_dir = base_path / class_name
        if not img_dir.exists():
            print(f"Warning: Folder {img_dir} does not exist, skipping.")
            continue

        # Get all images, shuffle, and take the first img_limit
        all_images = list(img_dir.iterdir())
        np.random.shuffle(all_images)
        selected_images = all_images[:img_limit]

        for img_path in selected_images:
            dd["images"].append(str(img_path))
            dd["labels"].append(class_name)

    return pd.DataFrame(dd)

# Define classes and paths
classes = ["Angry", "Fear", "Happy", "Sad", "Surprise"]
base_path = Path("./EmotionsArchive/Data").resolve()
output_path = Path("./EmotionsArchive/Data_1K").resolve()
output_path.mkdir(parents=True, exist_ok=True)

# Create dataframe
df = create_df_by_classes(base_path, classes, img_limit=1000)

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Copy the images to the new folder, preserving class subfolders
for class_name in classes:
    class_folder = output_path / class_name
    class_folder.mkdir(exist_ok=True)

df_grouped = df.groupby("labels")
for label, group in df_grouped:
    for img_path in group["images"]:
        dest_path = output_path / label / Path(img_path).name
        shutil.copy(img_path, dest_path)

print(f"Copied {len(df)} images to {output_path}")