import socket
import time
import math

def send_urscript_command(command, robot_ip="192.168.56.101", port=30002):
    """
    Sends a URScript command to the UR5 robot via socket communication.
    """
    try:
        # Open a socket connection
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((robot_ip, port))
        print(f"Connected to UR5 at {robot_ip}:{port}")
        
        # Send the URScript command
        s.sendall(command.encode('utf-8'))
        print(f"Sent command: {command}")

        # Wait for a while to ensure the command is executed
        time.sleep(1)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the socket connection
        s.close()
        print("Socket closed.")

def quaternion_to_rotation_vector(x, y, z, w):
    """
    Converts a quaternion to a rotation vector for URScript.
    """
    angle = 2 * math.acos(w)
    sin_angle = math.sqrt(1 - w * w)
    if sin_angle < 1e-6:
        return (x, y, z)  # when angle is close to 0, use direction
    return (x / sin_angle * angle, y / sin_angle * angle, z / sin_angle * angle)

if __name__ == "__main__":
    # Example: Move to joint positions [0, -1.57, 0, -1.57, 0, 0]

    ur_script = (
        "movej([3.1540732383728027, -1.5137341658221644, 1.337815761566162, -1.382540527974264, -1.5837348143206995, 4.678765296936035], a=0.5, v=0.3)\n"
    )
    
    # ur_script = (
    #     "movej([3.162419557571411, -1.5151379744159144, 1.3365941047668457, -1.3892954031573694, -1.5431402365313929, 3.1138763427734375], a=0.5, v=0.25)\n"
    # )
    
    # Update the robot IP address as needed
    send_urscript_command(ur_script, robot_ip="192.168.56.101") 

time.sleep(5)