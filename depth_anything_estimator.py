import os
import numpy as np
import requests
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

def main():
    # Load DepthAnything model
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

    image_directory = 'data_rgb/'
    new_image_directory_metric = 'DepthAnything_metric'
    new_image_directory_normalized = 'DepthAnything_normalized'

    # Loop through directory
    for dirpath, dirnames, filenames in os.walk(image_directory):
        print("*" * 10)
        print(f"dirpath: {dirpath}")
        print("*" * 10)
        for filename in filenames:
            if filename.endswith('.png'):
                # Prepare image path
                image_path = os.path.join(dirpath, filename)
                base_name, _ = os.path.splitext(filename)

                # Calculate subdirectory structure (train/test)
                sub_dir_structure = os.path.relpath(dirpath, image_directory)
                
                # Define paths for the normalized image and metric data
                normalized_dir_path = os.path.join(new_image_directory_normalized, sub_dir_structure)
                metric_dir_path = os.path.join(new_image_directory_metric, sub_dir_structure)
                
                normalized_image_path = os.path.join(normalized_dir_path, f"{base_name}_normalized.png")
                metric_image_path = os.path.join(metric_dir_path, f"{base_name}.pt")

                # Check if the output files already exist
                if os.path.exists(normalized_image_path) and os.path.exists(metric_image_path):
                    #print(f"Skipping {filename}, already processed.")
                    continue

                # Ensure output directories exist
                os.makedirs(normalized_dir_path, exist_ok=True)
                os.makedirs(metric_dir_path, exist_ok=True)

                # Load image
                image = Image.open(image_path).convert("RGB")

                # Generate DepthAnything depth data
                inputs = image_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth

                # Interpolate to original size
                depth_image = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )

                # Normalize the prediction
                output = depth_image.squeeze().cpu().numpy()
                formatted = (output * 255 / np.max(output)).astype("uint8")
                normalized_image = Image.fromarray(formatted)

                # Save the normalized image
                normalized_image.save(normalized_image_path, "PNG")
                print(f"Saved normalized depth image to {normalized_image_path}")

                # Save the un-normalized DepthAnything data to .pt file
                torch.save(predicted_depth, metric_image_path)
                print(f"Saved depth data from DepthAnything to {metric_image_path}")

main()
