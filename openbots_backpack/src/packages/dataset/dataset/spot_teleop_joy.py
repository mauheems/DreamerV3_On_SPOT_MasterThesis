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
    
  D-Pad (Axes 6,7):
    Up (+1 on Ax7)    -> Trot (Mode 2)
    Down (-1 on Ax7)  -> Crawl (Mode 4)
    Left (+1 on Ax6)  -> Auto (Mode 1)
    Right (-1 on Ax6) -> SpeedSelectTrot (Mode 3)
"""

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
        
        # D-Pad is often axes 6/7 (unused for gait now)
        self.axis_dpad_x = 6    # L/R
        self.axis_dpad_y = 7    # U/D
        
        # State
        self.locomotion_mode = 1 # Start at Auto
        self.is_sitting = False  # Track if robot is sitting
        self.was_moving = False  # Track if robot was moving last callback
        self.last_buttons = []
        self.last_axes = []
        
        # --- Publishers/Subscribers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # --- Service Clients ---
        self.sit_client = self.create_client(Trigger, '/sit')
        self.stand_client = self.create_client(Trigger, '/stand')
        self.locomotion_client = self.create_client(SetLocomotion, '/locomotion_mode')
        
        self.get_logger().info('Spot Joy Teleop Node Started. Waiting for /joy messages...')
        self.get_logger().info('Press X to Stand, O to Sit. L1/L2/R1/R2 select gaits.')

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
        
        # Update last state
        self.last_buttons = msg.buttons
        self.last_axes = msg.axes

    def call_trigger_service(self, client, name):
        if not client.service_is_ready():
            self.get_logger().warn(f'{name} service not not available!')
            return
        
        self.get_logger().info(f'Requesting: {name}...')
        future = client.call_async(Trigger.Request())
        # We don't block here with spin_until_future_complete because we are inside a callback
        future.add_done_callback(lambda future: self.service_response_callback(future, name))

    def set_locomotion(self, mode_id, mode_name):
        if not self.locomotion_client.service_is_ready():
            self.get_logger().warn('Locomotion service not available!')
            return
            
        self.get_logger().info(f'Switching Gait to: {mode_name} ({mode_id})')
        req = SetLocomotion.Request()
        req.locomotion_mode = mode_id
        future = self.locomotion_client.call_async(req)
        future.add_done_callback(lambda future: self.service_response_callback(future, f'Set Locomotive {mode_name}'))
    
    def service_response_callback(self, future, name):
        try:
            result = future.result()
            self.get_logger().info(f'{name} Result: {result.message} (Success: {result.success})')
        except Exception as e:
            self.get_logger().error(f'{name} Service call failed: {e}')
    
    def shutdown(self):
        """Safety shutdown: stop movement and sit robot."""
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
