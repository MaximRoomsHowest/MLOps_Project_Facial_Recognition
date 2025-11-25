import os
import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input images for a single class")
    parser.add_argument("--output_data", type=str, help="path to store resized images")
    parser.add_argument("--size", type=int, default=96, help="size to resize images to (size x size)")
    args = parser.parse_args()

    print("Input data:", args.data)
    print("Output folder:", args.output_data)

    os.makedirs(args.output_data, exist_ok=True)

    # Resize all images in the given folder
    for img_file in os.listdir(args.data):
        input_path = os.path.join(args.data, img_file)
        if not os.path.isfile(input_path):
            continue
        img = Image.open(input_path).convert("RGB")
        img_resized = img.resize((args.size, args.size))
        output_path = os.path.join(args.output_data, img_file)
        img_resized.save(output_path)

    print("Resizing completed.")

if __name__ == "__main__":
    main()