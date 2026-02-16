import socket
import rclpy
from rclpy.node import Node
from nmea_msgs.msg import Sentence

class NmeaUdpReceiver(Node):
    def __init__(self):
        super().__init__('nmea_udp_receiver')
        self.publisher_ = self.create_publisher(Sentence, 'nmea_sentence', 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', 10110))
        self.get_logger().info("Listening for NMEA sentences on UDP port 10110")
        self.timer = self.create_timer(0.05, self.read_udp)

    def read_udp(self):
        try:
            self.sock.settimeout(0.01)
            data, _ = self.sock.recvfrom(1024)
            sentence = data.decode('ascii').strip()
            msg = Sentence()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.sentence = sentence            
            self.publisher_.publish(msg)
        except socket.timeout:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = NmeaUdpReceiver()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
