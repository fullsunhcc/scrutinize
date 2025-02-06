# Open New terminal
# cd catkin_ws && source devel/setup.bash && roslaunch ur_calibration calibration_correction.launch robot_ip:=192.168.56.101 target_filename:="$(rospack find ur_calibration)/etc/ur5_example_calibration.yaml"
# Open New terminal
# cd catkin_ws && source devel/setup.bash && roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=192.168.56.101 noetic_config:=$(rospack find ur_calibration)/etc/ur5_example_calibration.yaml
# Open New terminal
# cd catkin_ws && source devel/setup.bash && roslaunch ur5_moveit_config moveit_planning_execution.launch limited:=true

import os
import threading

from unsloth import FastLanguageModel
from transformers import TextStreamer

from data_extractor import DataExtractor
from object_detection import ObjectDetection
from robot_control import RobotController
from data_extractor import DataExtractor
from llm_planner import LLMPlanner
from sgg_detection import SGGDetector
from link_prediction import LinkPrediction
from failure_detection import FailureDetection

def main():

    # print("Choose a task (1-5):")
    # print("1. Task_1: Block Stacking")
    # print("2. Task_2: Matching Block and Plate")
    # print("3. Task_3: Put objects on the desk")
    # print("4. Task_4: Cook the soup")
    # print("5. Task_5: Put in the microwave")
    # print("6. Exit")

    # while True:
    #     try:
    #         task_num = int(input("Enter your choice (1-5): "))
    #         if 1 <= task_num <= 6:
    #             break
    #         else:
    #             print("Invalid choice. Please select a number between 1 and 6.")
    #     except ValueError:
    #         print("Invalid input. Please enter a number between 1 and 6.")

    # if task_num == 6:
    #     print("Exiting program.")
    #     return

    task_num = 3  

    # Define the exp_num
    exp_num = 1
    exp_dir = f"./experiments/exp_{exp_num}"
    while os.path.exists(exp_dir):
        exp_num += 1
        exp_dir = f"./experiments/exp_{exp_num}"

    # Create the experiment directory
    os.makedirs(exp_dir)

    # Initialize components
    robot = RobotController(robot_ip="192.168.56.101", ur_port=30002, griper_port=63352)
    object = ObjectDetection(exp_num, task_num)
    data = DataExtractor(exp_num)
    llm = LLMPlanner(exp_num)
    detector = SGGDetector(exp_num, task_num)
    predictor = LinkPrediction(exp_num, gpu=-1)
    failure = FailureDetection()



    # Step 0: Object tracking 
    robot.extractor_start_motion()
    try:
        print("Starting robot motion and data extraction...")
        robot_thread = threading.Thread(target=robot.extractor_motion)
        data_thread = threading.Thread(target=data.start, args=(exp_num,))

        # Start both threads
        robot_thread.start()
        data_thread.start()

        # Wait for both threads to complete
        robot_thread.join()
        data.running = False  # Signal to stop data extraction
        data_thread.join()

        print("Robot motion and data extraction completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

    # Initialize robot position
    robot.origin()

    object.saved_track_bbox(exp_num, task_num)
    object_data = object.object_3d_coord(exp_num)
    object_list, llm_object_list = data.get_object_list(object_data)

    # Step 0: Task Planning
    instruction = "This is robot task planning. Give me the steps."
    wrong_result = ""
    fail_reason = ""

    task_input = "Command= Give me the step to place all objects on the desk."
    # task_input = input("Give me the task: ")

    while True:
        action_step, object_relation = llm.task_planning(instruction, wrong_result, fail_reason, task_input, llm_object_list)

        motions = llm.extract_(action_step)
        relations = llm.extract_(object_relation)
        
        k = len(motions)-1
        Task_GTSGG = FailureDetection.relations_to_GTSGG(relations, k)

        # Step 1: Get low-level action step
        for i in range(len(motions)):
            # # Update object_list BEFORE executing motion
            # object_list[:] = failure.get_object_list(entities, graph)[0]

            robot.execute_motion(motions[i], object_list, i)

        task_success = True

        if task_success:
            print("All tasks completed successfully.")
            break 

if __name__ == "__main__":
    main()