import os
import cv2
import torch
from PIL import Image

def main():

    #load ZoeDepth model
    repo = "isl-org/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(DEVICE)

    #loop through directory
    image_directory = 'data_rgb/'
    new_image_directory = 'ZoeDepth_data'

    for dirpath, dirnames, filenames in os.walk(image_directory):
            print("*"*10)
            print(f"dirpath: {dirpath}")
            print("*"*10)
            for filename in filenames:
                if filename.endswith('.png'):

                    #load image for ZoeDepth estimator
                    image_path = os.path.join(dirpath, filename)
                    check_exists_path = image_path.replace(image_directory, new_image_directory, 1 )

                    # if converting to unnormalized .pt file
                    base_name, _ = os.path.splitext(filename) 
                    depth_filename = base_name + ".pt"
                    check_exists_path = os.path.join(dirpath.replace(image_directory, new_image_directory, 1), depth_filename)


                    if os.path.exists(check_exists_path):
                        continue
                    else:
                        base_name, ext = os.path.splitext(filename)
                        image = Image.open(image_path).convert("RGB")

                        #generate ZoeDepth depth data
                        depth_image = zoe.infer_pil(image)

                        #save normalized ZoeDepth depth data to png
                        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        new_filename_normalized = base_name + ext
                        new_dir_path = dirpath.replace(image_directory, new_image_directory, 1 )
                        new_image_path_normalized = os.path.join(f"{new_dir_path}", new_filename_normalized)
                        os.makedirs(new_dir_path, exist_ok=True)
                        cv2.imwrite(new_image_path_normalized, normalized_depth)
                        print(f"Saved depth data from ZoeDepth to {new_image_path_normalized}")


                        #generate ZoeDepth depth data
                        depth_image = zoe.infer_pil(image, output_type="tensor")

                        #save un-normalized ZoeDepth depth data to pt
                        new_filename_un_normalized = base_name + ".pt"
                        new_dir_path = dirpath.replace(image_directory, new_image_directory, 1 )
                        new_image_path_un_normalized = os.path.join(f"{new_dir_path}", new_filename_un_normalized)
                        os.makedirs(new_dir_path, exist_ok=True)
                        torch.save(depth_image, new_image_path_un_normalized)
                        print(f"Saved depth data from ZoeDepth to {new_image_path_un_normalized}")

main()
