# Headless stitcher: fetch, stitch, and publish images as ROS Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import numpy as np
import cv2
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
from PIL import Image
import io
import time

class FrontImageStitcher(Node):
    def __init__(self, robot, jpeg_quality=50, publish_topic='/camera/frontmiddle_virtual/image', publish_rate=10):
        super().__init__('front_image_stitcher')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(RosImage, publish_topic, 10)
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.jpeg_quality = jpeg_quality
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)
        self.last_stamp = None

    def timer_callback(self):
        try:
            sources = ['frontright_fisheye_image', 'frontleft_fisheye_image']
            requests = [build_image_request(src, quality_percent=self.jpeg_quality) for src in sources]
            images = self.image_client.get_image(requests)
            img_dict = {}
            for img in images:
                if img.shot.image.format == image_pb2.Image.FORMAT_JPEG:
                    arr = np.asarray(Image.open(io.BytesIO(img.shot.image.data)))
                else:
                    self.get_logger().warn('Unsupported image format')
                    return
                img_dict[img.source.name] = arr
            if 'frontright_fisheye_image' in img_dict and 'frontleft_fisheye_image' in img_dict:
                # Simple horizontal stack (replace with better stitching if needed)
                stitched = np.hstack([img_dict['frontleft_fisheye_image'], img_dict['frontright_fisheye_image']])
                # Convert to ROS Image and publish
                msg = self.bridge.cv2_to_imgmsg(stitched, encoding='rgb8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'body'
                self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Error in stitcher: {e}')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Headless Spot front camera stitcher')
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-j', '--jpeg-quality-percent', type=int, default=50)
    parser.add_argument('--publish-topic', type=str, default='/camera/frontmiddle_virtual/image')
    parser.add_argument('--publish-rate', type=float, default=10.0)
    args = parser.parse_args()

    sdk = bosdyn.client.create_standard_sdk('front_cam_stitch')
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    rclpy.init()
    node = FrontImageStitcher(robot, jpeg_quality=args.jpeg_quality_percent, publish_topic=args.publish_topic, publish_rate=args.publish_rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
