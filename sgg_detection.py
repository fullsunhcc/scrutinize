import os
import re
import cv2
import sys
import torch
from SGG_Benchmark.demo.model import SGG_Model
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append("./SGG_Benchmark/")

class SGGDetector:
    def __init__(self, exp_num, task_num):
        self.exp_num = exp_num
        self.task_num = task_num
    
    def sgg_predict(self, exp_num, task_num):
        config_path = f"./SGG_Benchmark/checkpoints/task_{task_num}/config.yml"
        dict_path = f"./SGG_Benchmark/datasets/task_{task_num}/VG-SGG-dicts.json"
        weights_path = f"./SGG_Benchmark/checkpoints/task_{task_num}/best_model.pth"

        # Source directory containing images
        source_dir = f"./experiments/exp_{exp_num}/image_data/"

        # Output directory for results
        output_dir = f"./experiments/exp_{exp_num}/2dsgg_data/"

        # bbox_data directory 
        bbox_data_dir = f"./experiments/exp_{exp_num}/bbox_data/"

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(bbox_data_dir, exist_ok=True)

        # Get and sort the list of .jpg files
        file_list = sorted(
            [f for f in os.listdir(source_dir) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        # Initialize the model
        model = SGG_Model(config_path, dict_path, weights_path, tracking=True, rel_conf=0.8, box_conf=0.8, show_fps=False)

        # Process each image
        for filename in file_list:

            # Construct the image path
            img_path = os.path.join(source_dir, filename)
            
            # Read the image
            img = cv2.imread(img_path)
            img = cv2.flip(img, -1)
            # print("Processing:", img_path)

            # Predict and visualize
            img, graph, result, bbox_data = model.predict(img, img_path, visu=True)
            clean_img = model.nice_plot(img, graph)
            filename=filename.replace(".png",".txt") 

            # Save the result
            pil_image = Image.fromarray(clean_img)
            output_img_path = os.path.join(output_dir, filename.replace(".txt", ".png"))  # Use a valid image extension like .png

            # Save the image
            pil_image.save(output_img_path) 

            output_txt_path = os.path.join(output_dir, filename)
            with open(output_txt_path, 'w') as file:
                for sublist in result:
                    file.write(' '.join(map(str, sublist)) + '\n')

            bbox_data_txt_path = os.path.join(bbox_data_dir, filename)
            with open(bbox_data_txt_path, 'w') as file:
                for sublist in bbox_data:
                    file.write(', '.join(map(str, sublist)) + '\n')

        print("Processing completed. Results are saved")

