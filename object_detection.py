from ultralytics import YOLO
import cv2
import os
import re
from boxmot import DeepOcSort, OcSort
import numpy as np
import torch
from pathlib import Path
import random
import torch
import pyrealsense2 as rs
import time

class ObjectDetection:
    """
    Detect the bbox
    """
    def __init__(self, exp_num, yolo_model):
        self.exp_num = exp_num
        self.yolo_model = yolo_model
        self.track_colors = {}

    def get_random_color(self, track_id):
        if track_id not in self.track_colors:
            self.track_colors[track_id] = (
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
        return self.track_colors[track_id]

    def saved_2dsgg_bbox(self, exp_num, yolo_model):
        model = YOLO(f'./model/yolo_model/task_{yolo_model}.pt')

    def saved_track_bbox(self, exp_num, yolo_model):
        
        # Import YOLO model
        model = YOLO(f'./model/yolo_model/task_{yolo_model}.pt')

        # Define Device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Import Track model
        tracker = DeepOcSort(
                reid_weights = Path('./model/boxmot_model/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'),
                device = device,
                half = True, 
                per_class = True,
                det_thresh = 0.8,
                max_age = 65, # Maximum number of frames a track
                min_hits = 65, # Minimum number of frames a track
                iou_threshold = 0.2,
                delta_t = 10, # Temporal window for matching detections to tracks. Larger values allow associations across more frames but can increase computational cost
                asso_func = "giou",
                inertia = 0.2, # Controls the weight given to track's motion consistency when predicting future positions
                w_association_emb = 0.5, # Weight of embedding similarity in the combined association metric
                alpha_fixed_emb  = 0.2, # Smoothing factor for the fixed embedding updates in the tracker. Higher values give more weight to older embeddings.
                aw_param = 0.5, # Adaptive weighting parameter for combining different association metrics (IoU and embeddings). Higher values can skew associations toward embeddings.
                embedding_off = False, # disables the use of embeddings for association
                cmc_off = False, # disables the cascade matching cost (CMC) strategy, which can help with re-identifying objects
                aw_off = False, # disables adaptive weighting for association metrics
                Q_xy_scaling  = 0.001, # Scaling factor for the process noise in position (x, y) in the Kalman filter
                Q_s_scaling  = 0.001, # Scaling factor for the process noise in size (width, height) in the Kalman filter.
            )

        source_folder = f'./experiments/exp_{exp_num}/image_data/'
        bbox_path = f'./experiments/exp_{exp_num}/bbox_data/'
        sgg_path = f'./experiments/exp_{exp_num}/sgg_data/'
        os.makedirs(bbox_path, exist_ok=True)
        os.makedirs(sgg_path, exist_ok=True)

        file_list = sorted(
            [f for f in os.listdir(source_folder) if f.endswith(".png")],
            key=lambda x: int(os.path.splitext(x)[0])
        )

        # Define the class ID to label mapping
        label_class = {
            0: "red_block",
            1: "yellow_block",
            2: "blue_block",
            3: "red_plate",
            4: "yellow_plate",
            5: "blue_plate",
            6: "desk",
            7: "police_car",
            8: "ambulance",
            9: "pot",
            10: "carrot",
            11: "daikon",
            12: "cucumber",
            13: "microwave",
            14: "banana",
            15: "kiwi"
        }

        # Count
        bbox_number = 1
        sgg_number = 1

        # Main loop
        for image_file in file_list:
            img = cv2.imread(os.path.join(source_folder, image_file))
            if img is None:
                print(f"Warning: Unable to read image {image_file}. Skipping.")
                continue

            # Flip the image
            img = cv2.flip(img, -1)

            # Run object detection on the image
            results = model(img)

            # Extract detections for the tracker
            detections = []
            for result in results:
                for box in result.boxes:
                    bbox = box.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
                    bbox = bbox.flatten()
                    conf = float(box.conf.cpu().numpy())  # Confidence score
                    class_id = int(box.cls.cpu().numpy())  # Class ID

                    # Append detection: [x1, y1, x2, y2, confidence, class_id]
                    detections.append([bbox[0], bbox[1], bbox[2], bbox[3], conf, class_id])

            # Convert detections to NumPy array
            detections = np.array(detections) if detections else np.empty((0, 6))

            if detections.size == 0:
                print(f"No detections in {image_file}.")
                continue

            # Update the tracker with current frame detections
            tracks = tracker.update(detections, img)

            # Prepare the corresponding text file for this frame
            txt_output_file = os.path.join(bbox_path, f'{bbox_number}.txt')
            with open(txt_output_file, mode='w') as txt_file:
                # Write tracking details for the current frame
                for track in tracks:
                    track_id = int(track[4])  # Tracking ID
                    x1, y1, x2, y2 = map(int, track[:4])  # Bounding box coordinates
                    class_id = int(track[6])  # Class ID
                    class_label = label_class.get(class_id, "Unknown")  # Get the class label from the mapping

                    # Calculate center coordinates
                    center_x = (x1 + x2) / 2
                    center_x = 1280 - 1 - center_x 
                    center_y = (y1 + y2) / 2
                    center_y = 720 - 1 - center_y

                    # Write details to the text file
                    txt_file.write(
                        f"{class_label}, {track_id}, {center_x}, {center_y}\n"
                    )

                    # Get random color for the tracking ID
                    color = self.get_random_color(track_id)

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"ID: {track_id}, {class_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the annotated frame
            output_file = os.path.join(bbox_path, f'{bbox_number}.png')
            cv2.imwrite(output_file, img)
            bbox_number += 1

            txt_output_file = os.path.join(sgg_path, f'{sgg_number}.txt')
            with open(txt_output_file, mode='w') as txt_file:
                # Write tracking details for the current frame
                print (tracks)
                for track in tracks:
                    track_id = float(track[4])  # Tracking ID
                    x1, y1, x2, y2 = map(float, track[:4])  # Bounding box coordinates
                    pred = float(track[5]) # Predict Score
                    class_id = float(track[6])  # Class ID
                    class_label = label_class.get(class_id, "Unknown")  # Get the class label from the mapping

                    size = 640
                    x1 = size/1280 * x1
                    y1 = size/720 * y1
                    x2 = size/1280 * x2
                    y2 = size/720 * y2


                    # Write details to the text file
                    txt_file.write(
                        f"{x1}, {y1}, {x2}, {y2}, {pred}, {class_id}\n"
                    )

            # Save the annotated frame
            output_file = os.path.join(sgg_path, f'{sgg_number}.png')
            cv2.imwrite(output_file, img)
            sgg_number += 1



    @staticmethod
    def label_center_depth_data(exp_num, n):
        bbox_file = f'./experiments/exp_{exp_num}/bbox_data/{n}.txt'
        depth_file = f'./experiments/exp_{exp_num}/depth_data/{n}.txt'
        
        if not os.path.exists(bbox_file) or not os.path.exists(depth_file):
            print(f'Files {bbox_file} or {depth_file} do not exist.')
            return []
        
        # Read bbox data
        with open(bbox_file, 'r') as f:
            bbox_data = f.readlines()
        
        # Read depth data
        with open(depth_file, 'r') as f:
            depth_data = f.readlines()

        
        # Process bbox data
        result = []
        for line in bbox_data:
            parts = line.strip().split(',')
            label = parts[0]
            label_id = parts[1]
            center_x = float(parts[2])
            center_y = float(parts[3])
            x = round(center_x)
            y = round(center_y)
            depth_value = None
            for i, line in enumerate(depth_data):
                if i == y:
                    row = line.split()
                    if x < len(row):
                        depth_value = float(row[x]) 

            if depth_value is not None:
                if depth_value == 0.0:
                    print(f"{n}, {label}, {label_id}, {x}, {y}, depth value is 0.0")
                else: 
                    result.append(f'{label}, {label_id}, {x}, {y}, {depth_value}')
                    # print(f"{n}, {label}, {label_id}, {x}, {y}, {depth_value}")
            else:
                print("depth value not found") 
        
        return result
    
    @staticmethod
    def trans_rot_data(exp_num, n):
        file_path = f'./experiments/exp_{exp_num}/tf_data/{n}.txt'

        # Patterns to match translation and quaternion data
        translation_pattern = r"Translation:\s*\[([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\]"
        quaternion_pattern = r"Rotation:\s*in Quaternion\s*\[([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+),\s*([-+]?\d*\.\d+|\d+)\]"

        # Initialize the result list
        result = []
        found_translation = False
        found_quaternion = False

        # Open and read the file
        with open(file_path, 'r') as file:
            for line in file:
                # Debugging: Print each line being read
                # print(f"Reading line: {line.strip()}")

                # Search for translation data
                trans_match = re.search(translation_pattern, line)
                if trans_match and not found_translation:
                    result.extend([float(value) for value in trans_match.groups()])
                    found_translation = True
                    # print(f"Translation data matched: {result[:3]}")  # Debugging

                # Search for quaternion data
                quat_match = re.search(quaternion_pattern, line)
                if quat_match and not found_quaternion:
                    result.extend([float(value) for value in quat_match.groups()])
                    found_quaternion = True
                    # print(f"Quaternion data matched: {result[3:]}")  # Debugging

                # Stop once both translation and quaternion are found
                if found_translation and found_quaternion:
                    break

        if not found_translation:
            print("No translation data found in the file.")
        if not found_quaternion:
            print("No quaternion data found in the file.")

        return result

    @staticmethod    
    def track_first_last(exp_num, n_range):
        label_tracking = {}

        for n in n_range:
            label_center_depth = ObjectDetection.label_center_depth_data(exp_num, n)
            
            for entry in label_center_depth:
                try:
                    components = entry.split(',')
                    label = components[0].strip()
                    label_id = components[1].strip()

                    if label_id not in label_tracking:
                        # First occurrence of this label_id
                        label_tracking[label_id] = {'first': n, 'last': n}
                    else:
                        # Update the last occurrence of this label_id
                        label_tracking[label_id]['last'] = n

                except ValueError as e:
                    print(f"Error processing entry {entry}: {e}")
        
        return label_tracking
    
    def reset_camera(self):
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            # Delay to allow camera reset
            time.sleep(2)
        except RuntimeError as e:
            print(f"Camera reset failed: {e}")

    def object_3d_coord(self, exp_num):
        # Camera intrinsic parameters
        image_height = 720
        image_width = 1280
        cx = image_width / 2
        cy = image_height / 2
        fx = 644
        fy = 644

        ObjectDetection.reset_camera(self)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipeline.start(config)

        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = pipeline.wait_for_frames()

        # Aligning color frame to depth frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics


        n_range = range(1, 62)
        results = {}
        for n in n_range:
            label_center_depth = ObjectDetection.label_center_depth_data(exp_num, n)
            coord_quat = ObjectDetection.trans_rot_data(exp_num, n)
            for entry in label_center_depth:
                try:
                    t_x = coord_quat[0]
                    t_y = coord_quat[1]
                    t_z = coord_quat[2]
                    q_x = coord_quat[3]
                    q_y = coord_quat[4]
                    q_z = coord_quat[5]
                    q_w = coord_quat[6]

                    # Normalize the quaternion
                    norm = np.sqrt(q_x**2 + q_y**2 + q_z**2 + q_w**2)
                    q_x /= norm
                    q_y /= norm
                    q_z /= norm
                    q_w /= norm

                    # Compute the rotation matrix from the quaternion
                    R = np.array([
                        [1 - 2*(q_y**2 + q_z**2),     2*(q_x*q_y - q_z*q_w),       2*(q_x*q_z + q_y*q_w)],
                        [2*(q_x*q_y + q_z*q_w),       1 - 2*(q_x**2 + q_z**2),     2*(q_y*q_z - q_x*q_w)],
                        [2*(q_x*q_z - q_y*q_w),       2*(q_y*q_z + q_x*q_w),       1 - 2*(q_x**2 + q_y**2)]
                    ])

                    # Construct the 4x4 transformation matrix
                    T = np.eye(4)
                    T[0:3, 0:3] = R
                    T[0:3, 3] = [t_x, t_y, t_z]

                    # Split the entry into individual components
                    components = entry.split(',')
                    label = components[0].strip()
                    label_id = float(components[1].strip())
                    center_x = float(components[2].strip())
                    center_y = float(components[3].strip())
                    depth = float(components[4].strip())

                    dx, dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [center_x, center_y], depth)

                    coordinate = np.array([dx, dy, dz, 1])

                    transformed_coordinate = T @ coordinate
                    transformed_coordinate = transformed_coordinate[:3]
                    
                    # Store the results
                    key = (label, label_id)
                    if key not in results:
                        results[key] = []
                    results[key].append(transformed_coordinate)

                except ValueError as e:
                    print(f"Error processing entry {entry}: {e}")


        averages = {}
        for key, coords in results.items():
            averages[key] = np.mean(coords, axis=0)

        for key, avg_coord in averages.items():
            label, label_id = key

        # Example usage
        label_occurrences = ObjectDetection.track_first_last(exp_num, n_range)

        averages = {}
        for key, coords in results.items():
            averages[key] = np.mean(coords, axis=0)

        label_occurrences = ObjectDetection.track_first_last(exp_num, n_range)

        object_data = []
        for key, avg_coord in averages.items():
            label, label_id = key
            # Convert the float label_id to string for matching with occurrences
            label_id_str = str(int(label_id))  # Convert float to int first, then to string
            occurrences = label_occurrences.get(label_id_str, {"first": [], "last": []})
            object_data.append({
                "track_id": label_id_str,
                "label": label,
                "x": avg_coord[0], "y": avg_coord[1], "z": avg_coord[2],
                "s.v.t": occurrences['first'],
                "e.v.t": occurrences['last']
            })
            # print(f"Label ID: {label_id_str}, Label: {label}, Avg Coord: {avg_coord}, First n: {occurrences['first']}, Last k: {occurrences['last']}")

            class_label = {
                "red_block": 0,
                "yellow_block": 1,
                "blue_block": 2,
                "red_plate": 3,
                "yellow_plate": 4,
                "blue_plate": 5,
                "desk": 6,
                "police_car": 7,
                "ambulance": 8,
                "pot": 9,
                "carrot": 10,
                "daikon": 11,
                "cucumber": 12,
                "microwave": 13,
                "banana": 14,
                "kiwi": 15
            }

            # Path to the entities.txt file
            output_file = f"./experiments/exp_{exp_num}/entities.txt"

            # Open the file in write mode
            with open(output_file, "w") as file:
                
                # Write each entry in object_data
                for data in object_data:
                    # Extract values
                    label_id_str = data["track_id"]
                    label = class_label[data["label"]]
                    x = data["x"]
                    y = data["y"]
                    z = data["z"]
                    s_v_t = data["s.v.t"]
                    e_v_t = data["e.v.t"]
                    
                    # Format as a CSV-like row
                    row = f"{label_id_str}, {label}, {x}, {y}, {z}, {s_v_t}, {e_v_t}\n"
                    file.write(row)

        
        return object_data




