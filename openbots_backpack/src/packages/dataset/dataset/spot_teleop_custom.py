#!/usr/bin/env python3
"""
Custom teleop for Spot with discrete action space:
  Movement: w/a/s/d (forward/left/back/right)
  Rotation: q/e (rotate left/right)
  Gait: 1=Walk, 2=Trot, 3=Stair
  Posture: Space=Sit, z=Stand
  Exit: Ctrl+C
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger, SetBool
from spot_msgs.srv import SetLocomotion
import sys
import tty
import termios

class SpotTeleop(Node):
    def __init__(self):
        super().__init__('spot_teleop_custom')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Service clients
        self.sit_client = self.create_client(Trigger, '/sit')
        self.stand_client = self.create_client(Trigger, '/stand')
        self.locomotion_client = self.create_client(SetLocomotion, '/locomotion_mode')
        
        # Wait for services
        while not self.sit_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /sit service...')
        while not self.stand_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /stand service...')
        while not self.locomotion_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /locomotion_mode service...')
        
        # Gait mapping (Boston Dynamics LocomotionHint values)
        self.gaits = {
            '1': (1, 'Auto/Default'),      # HINT_AUTO
            '2': (2, 'Trot'),               # HINT_TROT
            '3': (3, 'SpeedSelectTrot'),    # HINT_SPEED_SELECT_TROT
            '4': (4, 'Crawl'),              # HINT_CRAWL
            '5': (10, 'SpeedSelectCrawl'),  # HINT_SPEED_SELECT_CRAWL
        }
        self.current_gait = 1

        # Current commanded velocities (to allow combined linear + angular)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        # Track robot state to prevent duplicate commands
        self.is_sitting = False
        self.is_standing = True  # Assume standing after power_on/stand
        
        self.get_logger().info('Spot Teleop Ready!')
        self.print_help()
        
    def print_help(self):
        help_text = """
╔═══════════════════════════════════════════════════════════════╗
║                 SPOT DISCRETE TELEOP CONTROLS                 ║
╠═══════════════════════════════════════════════════════════════╣
║  MOVEMENT (Discrete):                                         ║
║    W / A / S / D    - Forward / Left / Back / Right           ║
║    Q / E            - Rotate Left / Rotate Right              ║
║                                                               ║
║  GAIT SELECTION:                                              ║
║    1 - Auto/Default (1)    2 - Trot (2)                       ║
║    3 - SpeedSelectTrot (3) 4 - Crawl (4)                      ║
║    5 - SpeedSelectCrawl (10)                                  ║
║                                                               ║
║  POSTURE:                                                     ║
║    SPACE            - Sit                                     ║
║    Z                - Stand                                   ║
║                                                               ║
║  OTHER:                                                       ║
║    H                - Help                                    ║
║    Ctrl+C           - Exit & Stop                             ║
╚═══════════════════════════════════════════════════════════════╝
        """
        self.get_logger().info(help_text)
    
    def getch(self):
        """Read a single character from stdin without waiting for Enter."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def call_service(self, client, service_name):
        """Call a Trigger service."""
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done():
            try:
                response = future.result()
                status = "✓" if response.success else "✗"
                self.get_logger().info(f"{status} {service_name}: {response.message}")
                return response.success
            except Exception as e:
                self.get_logger().error(f"Service call failed: {e}")
                return False
        else:
            self.get_logger().error(f"Service call timed out: {service_name}")
            return False
    
    def publish_velocity(self, vx=0.0, vy=0.0, vz=0.0):
        """Publish a velocity command."""
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = vz
        self.cmd_vel_pub.publish(msg)

    def publish_current_velocity(self):
        """Publish the currently stored velocity command."""
        self.publish_velocity(self.vx, self.vy, self.vz)
    
    def set_locomotion_mode(self, mode_id, mode_name):
        """Call SetLocomotion service."""
        request = SetLocomotion.Request()
        request.locomotion_mode = mode_id
        future = self.locomotion_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done():
            try:
                response = future.result()
                status = "✓" if response.success else "✗"
                self.get_logger().info(f"{status} Gait → {mode_name} (ID: {mode_id})")
                self.current_gait = mode_id
            except Exception as e:
                self.get_logger().error(f"Failed to set locomotion mode: {e}")
        else:
            self.get_logger().error("Locomotion mode service timed out")
    
    def emergency_stop(self):
        """Emergency stop: stop movement and sit the robot."""
        self.get_logger().warn("🛑 EMERGENCY STOP - Sitting robot...")
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.publish_current_velocity()  # Stop all movement
        import time
        time.sleep(0.2)
        self.call_service(self.sit_client, "sit")
    
    def run(self):
        """Main control loop."""
        try:
            while True:
                key = self.getch().lower()
                
                # Movement commands
                if key == 'w':
                    self.vx = 0.5
                    self.publish_current_velocity()
                elif key == 's':
                    self.vx = -0.5
                    self.publish_current_velocity()
                elif key == 'a':
                    self.vy = 0.5
                    self.publish_current_velocity()
                elif key == 'd':
                    self.vy = -0.5
                    self.publish_current_velocity()
                elif key == 'q':
                    self.vz = 0.5
                    self.publish_current_velocity()
                elif key == 'e':
                    self.vz = -0.5
                    self.publish_current_velocity()
                
                # Gait selection (1-5)
                elif key in self.gaits:
                    mode_id, mode_name = self.gaits[key]
                    self.set_locomotion_mode(mode_id, mode_name)
                
                # Posture commands
                elif key == ' ':  # Space = sit
                    if self.is_sitting:
                        self.get_logger().info("Already sitting - command ignored")
                        continue
                    self.vx = 0.0
                    self.vy = 0.0
                    self.vz = 0.0
                    self.publish_current_velocity()  # Stop moving FIRST
                    import time
                    time.sleep(0.1)  # Brief pause to ensure stop command is processed
                    self.get_logger().info("→ Sitting...")
                    success = self.call_service(self.sit_client, "sit")
                    if success:
                        self.is_sitting = True
                        self.is_standing = False
                elif key == 'z':  # Z = stand
                    if self.is_standing:
                        self.get_logger().info("Already standing - command ignored")
                        continue
                    self.vx = 0.0
                    self.vy = 0.0
                    self.vz = 0.0
                    self.publish_current_velocity()  # Stop moving FIRST
                    import time
                    time.sleep(0.1)  # Brief pause
                    self.get_logger().info("→ Standing...")
                    success = self.call_service(self.stand_client, "stand")
                    if success:
                        self.is_sitting = False
                        self.is_standing = True
                
                # Other
                elif key == 'h':
                    self.print_help()
                elif key == '\x03':  # Ctrl+C
                    self.get_logger().info("Exit requested...")
                    break
                
        except KeyboardInterrupt:
            self.get_logger().info("KeyboardInterrupt received...")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")
        finally:
            # Always emergency stop on exit
            self.emergency_stop()

def main(args=None):
    rclpy.init(args=args)
    teleop = SpotTeleop()
    try:
        teleop.run()
    finally:
        teleop.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
