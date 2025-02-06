import os
import time
import subprocess
import threading
import pyrealsense2 as rs
import cv2
import numpy as np
import re

class DataExtractor:
    def __init__(self, exp_num):
        self.exp_num = exp_num
        self.running = True
        self.file_number = 0
    
    @staticmethod
    def reset_camera():
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            # Delay to allow camera reset
            time.sleep(2)
        except RuntimeError as e:
            print(f"Camera reset failed: {e}")

    def create_directory(self, base_dir, sub_dir):
        save_directory = f"./experiments/{base_dir}/{sub_dir}"
        os.makedirs(save_directory, exist_ok=True)
        return save_directory  # Return the directory path

    def combined_extractor(self, image_dir, depth_dir):
        fps = 30

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        profile = pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Get the depth Scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        frame_counter = 0
        
        time.sleep(4)
        try:
            while self.running:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_image = depth_image * depth_scale

                if frame_counter % fps == 0:
                    depth_filename = os.path.join(depth_dir, f'{self.second_counter + 1}.txt')
                    image_filename = os.path.join(image_dir, f'{self.second_counter + 1}.png')
                    cv2.imwrite(image_filename, color_image)
                    self.second_counter += 1
                    with open(depth_filename, 'w') as file:
                        for row in depth_image:
                            file.write(' '.join(map(str, row)) + '\n')

                frame_counter += 1

        finally:
            pipeline.stop()
            print("Depth and Camera extractor script ended")

    def tf_extractor(self, exp_num):
        frame_counter = 0
        fps = 30

        tf_dir = self.create_directory(base_dir=f"exp_{exp_num}", sub_dir="tf_data")

        try:
            while self.running:
                if frame_counter % fps == 0:
                    result = subprocess.run(
                                'timeout 1 rosrun tf tf_echo base tool0_controller 30 | head -n 5',
                                capture_output=True,
                                text=True,
                                shell=True
                            )
                    filename = os.path.join(tf_dir, f'{self.second_counter + 1}.txt')
                    with open(filename, 'w') as file:
                        file.write(result.stdout)
                frame_counter += 1

        finally:
            print("TF extractor script ended")

    def wait_for_termination(self):
        while self.running:
            if input().strip().lower() == 'q':
                self.running = False
                break

    def start(self, exp_num):
        self.running = True  # Reset the running flag
        self.second_counter = 0  # Reset the frame counter
        DataExtractor.reset_camera()
        image_dir = self.create_directory(base_dir=f"exp_{exp_num}", sub_dir="image_data")
        depth_dir = self.create_directory(base_dir=f"exp_{exp_num}", sub_dir="depth_data")

        combined_extractor_thread = threading.Thread(target=self.combined_extractor, args=(image_dir, depth_dir))
        tf_extractor_thread = threading.Thread(target=self.tf_extractor, args=(exp_num,))  # Pass exp_num here
        termination_thread = threading.Thread(target=self.wait_for_termination, daemon=True)

        combined_extractor_thread.start()
        tf_extractor_thread.start()
        time.sleep(1)
        termination_thread.start()
        time.sleep(1)
        combined_extractor_thread.join()
        tf_extractor_thread.join()

        print("Script ended")

    def get_object_list(self, object_data):

        label_mapping = {
            'red_block': 0,
            'yellow_block': 1,
            'blue_block': 2,
            'red_plate': 3,
            'yellow_plate': 4,
            'blue_plate': 5,
            'desk': 6,
            'police_car': 7,
            'ambulance': 8,
            'pot': 9,
            'carrot': 10,
            'daikon': 11,
            'cucumber': 12,
            'microwave': 13,
            'banana': 14,
            'kiwi': 15
        }

        label_counter = {}  # To keep track of occurrences of each label
        object_list = []
        object_names = []  # To store the string names of all objects
        
        for obj in object_data:
            label = obj['label']
            # Increment the counter for this label
            label_counter[label] = label_counter.get(label, 0) + 1
            
            # Create the unique object name using label and its occurrence count
            unique_object_name = f"{label}_{label_counter[label]}"
            
            # Append the object name to the object_names list
            object_names.append(unique_object_name)
            
            # Create the new dictionary for the object_list
            transformed_object = {
                'label': label_mapping[label],
                'object': unique_object_name,
                'x': obj['x'],
                'y': obj['y'],
                'z': obj['z']
            }
            
            # Add the transformed object to the object_list
            object_list.append(transformed_object)
        
        return object_list, f"Objects= {object_names}\n"
