import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import TwistStamped
from nmea_msgs.msg import Sentence

class VtgHeadingParser(Node):
    def __init__(self):
        super().__init__('vtg_heading_parser')
        self.sub = self.create_subscription(Sentence, '/nmea_sentence', self.callback, 10)
        self.deg_pub = self.create_publisher(Float64, '/heading_deg', 10)
        self.rad_pub = self.create_publisher(TwistStamped, '/heading_rad', 10)
        self.get_logger().info("VTG Heading parser active")

    def callback(self, msg):
        if msg.sentence.startswith('$GPVTG'):
            fields = msg.sentence.split(',')

            # Heading won't change if GPS is stationary (needs testing)
            if len(fields) > 1 and fields[1]:
                try:
                    heading_deg = float(fields[1])
                    heading_rad = math.radians(heading_deg)

                    # /heading_deg
                    deg_msg = Float64()
                    deg_msg.data = heading_deg
                    self.deg_pub.publish(deg_msg)

                    # /heading_rad
                    twist = TwistStamped()
                    twist.header.stamp = msg.header.stamp
                    twist.twist.angular.z = heading_rad
                    self.rad_pub.publish(twist)

                except ValueError:
                    self.get_logger().warn(f"Non-numeric heading in $GPVTG: {fields[1]}")
            else:
                self.get_logger().debug("No heading in $GPVTG message; likely due to GPS being stationary.")

def main(args=None):
    rclpy.init(args=args)
    node = VtgHeadingParser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
