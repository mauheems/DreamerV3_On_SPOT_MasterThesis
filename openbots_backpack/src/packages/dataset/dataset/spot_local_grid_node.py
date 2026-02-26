#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import time
import os
from threading import Lock, Thread, Event
import bosdyn.client
from bosdyn.client.local_grid import LocalGridClient
import cv2

class SpotLocalGridNode(Node):
    def __init__(self):
        super().__init__('spot_local_grid_node')

        self.declare_parameter('spot_ip', os.environ.get('SPOT_IP', '192.168.10.102'))
        self.declare_parameter('spot_username', os.environ.get('BOSDYN_CLIENT_USERNAME', 'user'))
        self.declare_parameter('spot_password', os.environ.get('BOSDYN_CLIENT_PASSWORD', 'corspotuser1'))
        self.declare_parameter('publish_frequency', 10.0)
        self.declare_parameter('grid_types', ['terrain'])  # Only terrain (obstacle is redundant)

        self.spot_ip = self.get_parameter('spot_ip').value
        self.username = self.get_parameter('spot_username').value
        self.password = self.get_parameter('spot_password').value
        self.publish_frequency = self.get_parameter('publish_frequency').value
        self.grid_types = self.get_parameter('grid_types').value

        self.bridge = CvBridge()
        self.publishers_dict = {}
        for gt in self.grid_types:
            topic_name = f'/spot/local_grid/{gt}'
            self.publishers_dict[gt] = self.create_publisher(Image, topic_name, 10)

        self.spot_client = None
        self.local_grid_client = None
        self.stop_event = Event()
        self.thread = None

        self.initialize_spot_client()

    def initialize_spot_client(self):
        try:
            self.get_logger().info(f"Connecting to SPOT at {self.spot_ip}...")
            sdk = bosdyn.client.create_standard_sdk('spot_local_grid_node')
            self.spot_client = sdk.create_robot(self.spot_ip)
            self.spot_client.authenticate(self.username, self.password)
            self.spot_client.sync_with_directory()
            self.spot_client.time_sync.wait_for_sync()
            self.local_grid_client = self.spot_client.ensure_client(LocalGridClient.default_service_name)
            self.get_logger().info("✅ Connected to SPOT Local Grid Service")

            self.thread = Thread(target=self.fetch_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            self.get_logger().error(f"Failed to connect to SPOT: {e}")

    def fetch_loop(self):
        interval = 1.0 / self.publish_frequency
        while not self.stop_event.is_set():
            loop_start = time.time()
            try:
                responses = self.local_grid_client.get_local_grids(self.grid_types)
                for response in responses:
                    gt_name = response.local_grid_type_name
                    if gt_name in self.publishers_dict:
                        grid_data = self._unpack_grid(response.local_grid)
                        grid_2d = grid_data.reshape(
                            response.local_grid.extent.num_cells_y,
                            response.local_grid.extent.num_cells_x
                        ).astype(np.float32)
                        
                        # Publish as 32FC1 Image to preserve float precision
                        msg = self.bridge.cv2_to_imgmsg(grid_2d, encoding="32FC1")
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.header.frame_id = "body" # Usually grid is relative to body
                        self.publishers_dict[gt_name].publish(msg)

            except Exception as e:
                self.get_logger().warn(f"Error fetching grids: {e}")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, interval - elapsed))

    def _unpack_grid(self, grid_proto):
        expected_size = grid_proto.extent.num_cells_x * grid_proto.extent.num_cells_y
        data_bytes = len(grid_proto.data)
        rle_count_sum = sum(grid_proto.rle_counts) if grid_proto.rle_counts else 0
        
        if rle_count_sum == expected_size and len(grid_proto.rle_counts) > 0:
            num_rle_entries = len(grid_proto.rle_counts)
            bytes_per_entry = data_bytes / num_rle_entries
            if abs(bytes_per_entry - 1.0) < 0.1:
                data_type = np.int8
            elif abs(bytes_per_entry - 2) < 0.1:
                data_type = np.int16
            elif abs(bytes_per_entry - 4) < 0.1:
                data_type = np.float32
            else:
                data_type = np.int8
            cells_data = np.frombuffer(grid_proto.data, dtype=data_type)
            full_grid = np.repeat(cells_data[:len(grid_proto.rle_counts)], grid_proto.rle_counts).astype(data_type)
        else:
            if data_bytes == expected_size * 2:
                data_type = np.int16
            elif data_bytes == expected_size * 4:
                data_type = np.float32
            else:
                data_type = np.int16
            full_grid = np.frombuffer(grid_proto.data, dtype=data_type)
        
        full_grid_float = full_grid.astype(np.float64)
        if grid_proto.cell_value_scale != 0:
            full_grid_float *= grid_proto.cell_value_scale
        full_grid_float += grid_proto.cell_value_offset
        return full_grid_float

def main(args=None):
    rclpy.init(args=args)
    node = SpotLocalGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        if node.thread:
            node.thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
