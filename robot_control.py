import math
import socket
import time

class RobotController:
    """
    Manages the robot's low-level motions.
    """
    def __init__(self, robot_ip, ur_port, griper_port):
        self.robot_ip = robot_ip
        self.ur_port = ur_port
        self.griper_port = griper_port

    @staticmethod
    def quat_to_rot(x, y, z, w):
        """
        Converts a quaternion to a rotation vector for URScript.
        """
        norm = math.sqrt(x**2 + y**2 + z**2 + w**2)
        if norm == 0:
            raise ValueError("Quaternion cannot be zero.")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        angle = 2 * math.acos(w)
        sin_angle = math.sqrt(1 - w**2)
        if sin_angle < 1e-6:
            return (0.0, 0.0, 0.0)

        rx = x / sin_angle * angle
        ry = y / sin_angle * angle
        rz = z / sin_angle * angle

        return (rx, ry, rz)

    def send_urscript_command(self, command):
        """
        Sends a URScript command to the UR5 robot via socket communication.
        """
        try:
            # Open socket connection
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.robot_ip, self.ur_port))
                print(f"Connected to UR5 at {self.robot_ip}:{self.ur_port}")
                
                # Set a default command if "end" is specified
                if command == "end":
                    command = (
                        "movej([3.1540732383728027, -1.5137341658221644, 1.337815761566162, -1.382540527974264, -1.5837348143206995, 4.678765296936035], a=0.5, v=0.4)\n"
                    )

                # Send the command
                s.sendall(command.encode('utf-8'))
                print(f"Sent command: {command}")
        except Exception as e:
            print(f"Error: {e}")

    def control_gripper(self, action):
        """
        Controls the gripper by sending commands via socket.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.robot_ip, self.griper_port))
                if action == "close":
                    s.sendall(b'SET POS 250\n')  # Adjust gripper command
                elif action == "open":
                    s.sendall(b'SET POS 0\n')   # Adjust gripper command
                else:
                    raise ValueError(f"Gripper action '{action}' not recognized.")
        except Exception as e:
            print(f"Error controlling gripper: {e}")

    def extractor_start_motion(self):
        self.send_urscript_command("movej([3.16207218170166, -2.759742085133688, 1.7633061408996582, -1.4272635618792933, -1.560467545186178, 3.113457202911377], a=0.5, v=0.3)\n")
        time.sleep(10)

    def extractor_motion(self):
        time.sleep(10)
        self.send_urscript_command("movej([2.306004524230957, -1.3515437285052698, 0.7563061714172363, -0.5530074278460901, -1.062127415333883, 0.6450895071029663], a=0.5, v=0.125)\n")
        time.sleep(20)
        self.send_urscript_command("movej([3.104856252670288, -1.0770905653582972, 0.4747905731201172, -0.6416338125811976, -1.5775187651263636, -0.07232410112489873], a=0.5, v=0.045)\n")
        time.sleep(20)
        self.send_urscript_command("movej([3.7378342151641846, -1.0482152144061487, 0.35896778106689453, -0.6064284483539026, -2.0163548628436487, -0.9735849539386194], a=0.5, v=0.045)\n")
        time.sleep(20)
        self.send_urscript_command("movej([4.918083190917969, -2.170814816151754, 1.6273512840270996, -0.8259504477130335, -2.4163408915149134, -1.3369219938861292], a=0.5, v=0.07)\n")
        time.sleep(20)


    def origin(self):
        self.send_urscript_command("end")
        time.sleep(8)

    def execute_motion(self, motion, object_list, i):
        """
        Executes predefined robot motions.
        """
        if motion['method'] == 'pick_and_place':
            name_a = motion['arguments']['object_a']
            name_b = motion['arguments']['object_b']

            # Find object_a and object_b in the object_list
            object_a = next((obj for obj in object_list if obj['object'] == name_a), None)
            object_b = next((obj for obj in object_list if obj['object'] == name_b), None)

            ### Task_1 ###
            if object_a["label"] in [0, 1, 2] and object_b["label"] in [0, 1, 2]:

                # Extract position information
                a_x, a_y, a_z = object_a["x"], object_a["y"], object_a["z"]
                b_x, b_y, b_z = object_b["x"], object_b["y"], object_b["z"]

                # Threshold for position
                # object_a
                a_x = a_x + 0.01
                a_y = a_y 
                a_z = a_z + 0.1
                # object_b
                b_x = b_x + 0.01
                b_y = b_y 
                b_z = b_z + 0.15

                # Fixed quaternion for rotation
                qx, qy, qz, qw = 0.9996570492766498, 0.024216607503053426, -0.006214965322989974, 0.0077919162328075765
                rx, ry, rz = self.quat_to_rot(qx, qy, qz, qw)

                # Move to pick position
                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, {a_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                # Close the gripper
                self.control_gripper("close")
                time.sleep(4)

                # Move to place position
                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, {b_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)
                
                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)


                # Open the gripper
                self.control_gripper("open")
                time.sleep(4)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(4)
                
                # End the sub-task
                self.send_urscript_command("end")
                time.sleep(4)

            ### Task_2 ###
            elif object_a["label"] in [0, 1, 2] and object_b["label"] in [3, 4, 5]: 

                # Extract position information
                a_x, a_y, a_z = object_a["x"], object_a["y"], object_a["z"]
                b_x, b_y, b_z = object_b["x"], object_b["y"], object_b["z"]

                # Threshold for position
                # object_a
                a_x = a_x + 0.01 
                a_y = a_y
                a_z = a_z + 0.1
                # object_b
                b_x = b_x + 0.01
                b_y = b_y
                b_z = b_z + 0.155

                # Fixed quaternion for rotation
                qx, qy, qz, qw = 0.9996570492766498, 0.024216607503053426, -0.006214965322989974, 0.0077919162328075765
                rx, ry, rz = self.quat_to_rot(qx, qy, qz, qw)

                # Move to pick position
                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, {a_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                # Close the gripper
                self.control_gripper("close")
                time.sleep(4)

                # Move to place position
                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, {b_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)


                # Open the gripper
                self.control_gripper("open")
                time.sleep(4)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(4)
                
                # End the sub-task
                self.send_urscript_command("end")
                time.sleep(4)

            ### Task_3 ###
            elif object_a["label"] in [7, 8] and object_b["label"] in [6]:

                # Extract position information
                a_x, a_y, a_z = object_a["x"], object_a["y"], object_a["z"]
                b_x, b_y, b_z = object_b["x"], object_b["y"], object_b["z"]

                # Threshold for position
                # object_a
                a_x = a_x + 0.01

                if object_a["label"] == 8:
                    a_y = a_y + 0.01
                    a_z = a_z + 0.1 - 0.01
                
                else:
                    a_y = a_y
                    a_z = a_z + 0.1

                # object_b
                if object_a["label"] == 7:
                    b_x = b_x + 0.11

                else:
                    b_x = b_x - 0.09
                
                b_y = b_y
                b_z = b_z + 0.15

                # Fixed quaternion for rotation
                qx, qy, qz, qw = 0.9996570492766498, 0.024216607503053426, -0.006214965322989974, 0.0077919162328075765
                rx, ry, rz = self.quat_to_rot(qx, qy, qz, qw)

                # Move to pick position
                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, {a_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                # Close the gripper
                self.control_gripper("close")
                time.sleep(4)

                # Move to place position
                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, {b_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)


                # Open the gripper
                self.control_gripper("open")
                time.sleep(4)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(4)
                
                # End the sub-task
                self.send_urscript_command("end")
                time.sleep(4)

            ### Task_4 ###
            elif object_a["label"] in [10, 11, 12] and object_b["label"] in [9]:

                # Extract position information
                a_x, a_y, a_z = object_a["x"], object_a["y"], object_a["z"]
                b_x, b_y, b_z = object_b["x"], object_b["y"], object_b["z"]

                # Threshold for position
                # object_a
                a_x = a_x + 0.01
                a_y = a_y
                if object_a["label"] == 10:
                    a_z = a_z + 0.115
                else:
                    a_z = a_z + 0.1
                # object_b
                b_x = b_x + 0.01
                b_y = b_y 
                b_z = b_z + 0.25

                # Fixed quaternion for rotation
                qx, qy, qz, qw = 0.9996570492766498, 0.024216607503053426, -0.006214965322989974, 0.0077919162328075765
                rx, ry, rz = self.quat_to_rot(qx, qy, qz, qw)

                # Move to pick position
                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, {a_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                # Close the gripper
                self.control_gripper("close")
                time.sleep(4)

                # Move to place position
                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, {b_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)


                # Open the gripper
                self.control_gripper("open")
                time.sleep(4)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(4)
                
                # End the sub-task
                self.send_urscript_command("end")
                time.sleep(4)

        elif motion['method'] == 'remove' and i == 0:
            name_a = motion['arguments']['object_a']

            # Find object_a and object_b in the object_list
            object_a = next((obj for obj in object_list if obj['object'] == name_a), None)

            if object_a["label"] == 0 or 1 or 2: 

                # Extract position information
                a_x, a_y, a_z = object_a["x"], object_a["y"], object_a["z"]

                # Threshold for position
                # object_a
                a_x = a_x 
                a_y = a_y
                a_z = a_z + 0.1
                # object_b
                b_x = 0.2508
                b_y = 0.2024
                b_z = 0.145

                # Fixed quaternion for rotation
                qx, qy, qz, qw = 0.9996570492766498, 0.024216607503053426, -0.006214965322989974, 0.0077919162328075765
                rx, ry, rz = self.quat_to_rot(qx, qy, qz, qw)

                # Move to pick position
                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, {a_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                # Close the gripper
                self.control_gripper("close")
                time.sleep(4)

                # Move to place position
                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, {b_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                
                # Open the gripper
                self.control_gripper("open")
                time.sleep(4)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(4)
                
                # End the sub-task
                self.send_urscript_command("end")
            time.sleep(4)


        elif motion['method'] == 'remove' and i == 1:
            name_a = motion['arguments']['object_a']

            # Find object_a and object_b in the object_list
            object_a = next((obj for obj in object_list if obj['object'] == name_a), None)

            if object_a["label"] == 0 or 1 or 2: 

                # Extract position information
                a_x, a_y, a_z = object_a["x"], object_a["y"], object_a["z"]

                # Threshold for position
                # object_a
                a_x = a_x 
                a_y = a_y
                a_z = a_z + 0.1
                # object_b
                b_x = 0.2508
                b_y = 0.0294
                b_z = 0.145

                # Fixed quaternion for rotation
                qx, qy, qz, qw = 0.9996570492766498, 0.024216607503053426, -0.006214965322989974, 0.0077919162328075765
                rx, ry, rz = self.quat_to_rot(qx, qy, qz, qw)

                # Move to pick position
                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{a_x}, {a_y}, {a_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                # Close the gripper
                self.control_gripper("close")
                time.sleep(4)

                # Move to place position
                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, {b_z}, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(5)

                
                # Open the gripper
                self.control_gripper("open")
                time.sleep(4)

                self.send_urscript_command(f"movep(p[{b_x}, {b_y}, 0.5, {rx}, {ry}, {rz}], a=0.1, v=0.2)\n")
                time.sleep(4)
                
                # End the sub-task
                self.send_urscript_command("end")
            time.sleep(4)

        else:
            raise ValueError(f"Motion '{motion}' not defined.")
