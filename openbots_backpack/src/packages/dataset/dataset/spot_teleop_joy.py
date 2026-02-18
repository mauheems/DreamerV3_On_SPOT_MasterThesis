#!/usr/bin/env python3
"""
Spot Teleop Node for PS4 Controller (DualShock 4)
Requires: ros2 run joy joy_node

Mapping (Standard Linux Input):
  Axes:
    Left Stick (0,1)  -> Linear X (Fwd/Back), Linear Y (Left/Right)
    Right Stick (3)   -> Angular Z (Yaw)
    
    Buttons:
        L1 (4)            -> Auto (Mode 1)
        L2 (6)            -> SpeedSelectTrot (Mode 3)
        R1 (5)            -> Crawl (Mode 4)
        R2 (7)            -> Jog (Mode 7)
        L3 (10)           -> SpeedSelectAmble (Mode 6)
        X (0)             -> Stand
        Circle (1)        -> Sit
        Square (2)        -> Start/Stop Recording
        Triangle (3)      -> Delete/Discard Recording
    
  D-Pad (Axes 6,7):
    Up (+1 on Ax7)    -> Trot (Mode 2)
    Down (-1 on Ax7)  -> Crawl (Mode 4)
    Left (+1 on Ax6)  -> Auto (Mode 1)
    Right (-1 on Ax6) -> SpeedSelectTrot (Mode 3)
"""

import os
import signal
import subprocess
import shutil
import time
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
from spot_msgs.srv import SetLocomotion

class SpotJoyTeleop(Node):
    def __init__(self):
        super().__init__('spot_joy_teleop')

        # --- Parameters ---
        # Scaling factors
        self.linear_scale = 1.0   # Max m/s
        self.angular_scale = 1.0  # Max rad/s
        
        # Button/Axis Mappings (PS4 Standard)
        self.axis_linear_x = 1  # Left Stick U/D
        self.axis_linear_y = 0  # Left Stick L/R
        self.axis_angular_z = 3 # Right Stick L/R
        
        # Gait buttons (PS4 Standard: L1=4, R1=5, L2=6, R2=7, L3=10)
        self.btn_gait_auto = 4  # L1 -> Auto (1)
        self.btn_gait_speed_trot = 6  # L2 -> SpeedSelectTrot (3)
        self.btn_gait_crawl = 5  # R1 -> Crawl (4)
        self.btn_gait_jog = 7  # R2 -> Jog (7)
        self.btn_gait_speed_amble = 10  # L3 -> SpeedSelectAmble (6)
        self.btn_stand = 0      # X
        self.btn_sit = 1        # Circle
        self.btn_record_toggle = 2  # Square -> Start/Stop Recording
        self.btn_record_delete = 3  # Triangle -> Delete/Discard Recording
        
        # D-Pad is often axes 6/7 (unused for gait now)
        self.axis_dpad_x = 6    # L/R
        self.axis_dpad_y = 7    # U/D
        
        # State
        self.locomotion_mode = 1 # Start at Auto
        self.is_sitting = False  # Track if robot is sitting
        self.was_moving = False  # Track if robot was moving last callback
        self.last_buttons = []
        self.last_axes = []
        
        # Recording state
        self.recording_process = None
        self.is_recording = False
        self.current_bag_name = None
        self.output_dir = "/home/ob/openbots_ws/src/packages/dataset/recorded_data"
        self.recorder_script = "/home/ob/openbots_ws/src/packages/dataset/episode_bag_recorder.py"
        
        # --- Publishers/Subscribers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # --- Service Clients ---
        self.sit_client = self.create_client(Trigger, '/sit')
        self.stand_client = self.create_client(Trigger, '/stand')
        self.locomotion_client = self.create_client(SetLocomotion, '/locomotion_mode')
        
        self.get_logger().info('Spot Joy Teleop Started - Square: record, Triangle: discard')

    def joy_callback(self, msg):
        # Initialize last state if empty
        if not self.last_buttons:
            self.last_buttons = msg.buttons
            self.last_axes = msg.axes
            return

        # --- Movement (Twist) ---
        twist = Twist()
        # Map axes to twist
        # Note: Joystick Up is usually negative (-1.0), so we negate it for Forward (+X)
        twist.linear.x = float(msg.axes[self.axis_linear_x]) * self.linear_scale 
        twist.linear.y = float(msg.axes[self.axis_linear_y]) * self.linear_scale
        # Right stick left/right controls yaw; up/down is ignored (axis 3 is horizontal)
        twist.angular.z = float(msg.axes[self.axis_angular_z]) * self.angular_scale
        
        # Snap-to-forward zone: if moving mostly forward, ignore small lateral inputs
        # to prevent joystick jitter from causing instability
        # Tune these values to adjust the forward cone:
        #   - forward_threshold: increase to make cone narrower (e.g., 0.7, 0.8)
        #   - lateral_threshold: decrease to make cone narrower (e.g., 0.1, 0.15)
        forward_threshold = 0.6   # Only apply snap if forward command > this
        lateral_threshold = 0.2   # Only snap if lateral command < this
        if abs(twist.linear.x) > forward_threshold and abs(twist.linear.y) < lateral_threshold:
            twist.linear.y = 0.0  # Snap lateral movement to zero
        
        # Check if any movement is commanded
        has_movement = abs(twist.linear.x) > 0.01 or abs(twist.linear.y) > 0.01 or abs(twist.angular.z) > 0.01
        
        # Resume from sitting if movement commanded
        if self.is_sitting and has_movement:
            self.is_sitting = False
            self.get_logger().info('Movement detected - resuming from sit')
        
        self.was_moving = has_movement
        
        # Only publish cmd_vel if not sitting
        if not self.is_sitting:
            self.cmd_vel_pub.publish(twist)

        # --- Service Calls (Edge Detection) ---
        
        # Stand (X)
        if msg.buttons[self.btn_stand] and not self.last_buttons[self.btn_stand]:
            self.is_sitting = False
            self.call_trigger_service(self.stand_client, 'Stand')
            
        # Sit (Circle) - stop movement first
        if msg.buttons[self.btn_sit] and not self.last_buttons[self.btn_sit]:
            self.cmd_vel_pub.publish(Twist())  # Send zero velocity
            import time
            time.sleep(0.2)  # Brief delay
            self.is_sitting = True
            self.call_trigger_service(self.sit_client, 'Sit')
            
        # --- Gait Selection (Buttons) ---
        if msg.buttons[self.btn_gait_auto] and not self.last_buttons[self.btn_gait_auto]:
            self.set_locomotion(1, 'Auto')
        if msg.buttons[self.btn_gait_speed_trot] and not self.last_buttons[self.btn_gait_speed_trot]:
            self.set_locomotion(3, 'SpeedSelectTrot')
        if msg.buttons[self.btn_gait_crawl] and not self.last_buttons[self.btn_gait_crawl]:
            self.set_locomotion(4, 'Crawl')
        if msg.buttons[self.btn_gait_jog] and not self.last_buttons[self.btn_gait_jog]:
            self.set_locomotion(7, 'Jog')
        if msg.buttons[self.btn_gait_speed_amble] and not self.last_buttons[self.btn_gait_speed_amble]:
            self.set_locomotion(6, 'SpeedSelectAmble')
        
        # --- Recording Controls ---
        # Square: Start/Stop recording
        if msg.buttons[self.btn_record_toggle] and not self.last_buttons[self.btn_record_toggle]:
            self.toggle_recording()
        
        # Triangle: Delete/Discard recording
        if msg.buttons[self.btn_record_delete] and not self.last_buttons[self.btn_record_delete]:
            self.discard_recording()
        
        # Update last state
        self.last_buttons = msg.buttons
        self.last_axes = msg.axes

    def call_trigger_service(self, client, name):
        if not client.service_is_ready():
            self.get_logger().error(f'{name} service not available!')
            return
        
        self.get_logger().info(f'{name}')
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda future: self.service_response_callback(future, name))

    def set_locomotion(self, mode_id, mode_name):
        if not self.locomotion_client.service_is_ready():
            self.get_logger().error('Locomotion service not available!')
            return
            
        self.get_logger().info(f'Gait: {mode_name}')
        req = SetLocomotion.Request()
        req.locomotion_mode = mode_id
        future = self.locomotion_client.call_async(req)
        future.add_done_callback(lambda future: self.service_response_callback(future, f'Gait {mode_name}'))
    
    def service_response_callback(self, future, name):
        try:
            result = future.result()
            if not result.success:
                self.get_logger().error(f'{name} failed: {result.message}')
        except Exception as e:
            self.get_logger().error(f'{name} error: {e}')
    
    def toggle_recording(self):
        """Start or stop recording via joystick."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start a new recording episode."""
        if self.is_recording:
            self.get_logger().warn('Already recording!')
            return
        
        # Generate bag name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_bag_name = f"spot_bag_{timestamp}_joystick_collected"
        
        self.get_logger().info(f'Recording started: {self.current_bag_name}')
        
        topics = [
            "/camera/frontmiddle_virtual/image",
            "/depth_registered/frontleft/image",
            "/depth_registered/frontright/image",
            "/odometry",
            "/cmd_vel",
            "/status/mobility_params",
            "/spot/local_grid/terrain",
            "/spot/local_grid/obstacle_distance",
            "/tf",
            "/tf_static",
        ]
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            cmd = ["ros2", "bag", "record", "--output", os.path.join(self.output_dir, self.current_bag_name)] + topics
            self.recording_process = subprocess.Popen(cmd)
            self.is_recording = True
        except Exception as e:
            self.get_logger().error(f'Failed to start recording: {e}')
            self.is_recording = False
    
    def stop_recording(self):
        """Stop and save the current recording."""
        if not self.is_recording:
            self.get_logger().warn('No active recording!')
            return
        
        self.get_logger().info(f'Recording saved: {self.current_bag_name}')
        
        if self.recording_process and self.recording_process.poll() is None:
            self.recording_process.send_signal(signal.SIGINT)
            try:
                self.recording_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.recording_process.kill()
        
        self.is_recording = False
        self.current_bag_name = None
    
    def discard_recording(self):
        """Stop recording and delete the episode without saving."""
        if not self.is_recording:
            self.get_logger().warn('No active recording to discard!')
            return
        
        self.get_logger().info(f'Recording discarded: {self.current_bag_name}')
        
        # Stop the recording process
        if self.recording_process and self.recording_process.poll() is None:
            self.recording_process.send_signal(signal.SIGINT)
            try:
                self.recording_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.recording_process.kill()
        
        # Delete the bag directory
        if self.current_bag_name:
            bag_path = os.path.join(self.output_dir, self.current_bag_name)
            if os.path.exists(bag_path):
                shutil.rmtree(bag_path)
        
        self.is_recording = False
        self.current_bag_name = None
    
    def shutdown(self):
        """Safety shutdown: stop movement, sit robot, and stop recording."""
        if self.is_recording:
            self.stop_recording()
        print('\n⚠️  Teleop shutting down - Please manually sit the robot (press Circle/O)')
        print('   Or use: ros2 service call /sit std_srvs/srv/Trigger')

def main(args=None):
    rclpy.init(args=args)
    node = SpotJoyTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
