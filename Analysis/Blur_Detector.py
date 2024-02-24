import os
import numpy as np
import blur_detector
import cv2

def compute_average_blurriness_per_class(data_path):
    class_blur_scores = {}

    for class_name in os.listdir(data_path):
        class_folder = os.path.join(data_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        blur_scores = []
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path, 0)

            if img is not None:
                blur_map = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
                blur_score = np.mean(blur_map)
                blur_scores.append(blur_score)

        if blur_scores:
            average_blur = np.mean(blur_scores)
            class_blur_scores[class_name] = average_blur

    return class_blur_scores


data_path = "/Users/manuelfreistein/Desktop/Masterarbeit/Data/raw_data/topex-printer/test"
class_blur_scores = compute_average_blurriness_per_class(data_path)
for class_name, blur_score in class_blur_scores.items():
    print(f"{class_name}: {blur_score}")
