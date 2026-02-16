#!/usr/bin/env python3
"""
Convert SPOT Rosbags to DreamerV3-compatible HDF5 episodes.
Uses 'cv_bridge' and 'rclpy' serialization tools to parse bag data offline.
"""

import argparse
import os
import sys
import numpy as np
import h5py
import cv2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from cv_bridge import CvBridge

# Message types import (adjust if needed for custom msgs)
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

DEFAULT_RECORD_DIR = "/home/ob/openbots_ws/src/packages/dataset/recorded_data"

def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options

class BagConverter:
    def __init__(self, bag_path, output_dir, target_freq=5.0):
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.target_freq = target_freq
        self.bridge = CvBridge()
        
        # Buffer for latest messages
        self.latest = {
            '/camera/frontleft/image': None,
            '/camera/frontright/image': None,
            '/camera/frontmiddle_virtual/image': None,
            '/depth/frontleft/image': None,
            '/depth/frontright/image': None,
            '/spot/local_grid/terrain': None,
            '/spot/local_grid/obstacle_distance': None,
            '/odometry': None,
            '/cmd_vel': None,
            '/status/mobility_params': None
        }
        
        # Episode storage
        self.episode = {
            'images_left': [],
            'images_right': [],
            'images_front': [],
            'depth_left': [],
            'depth_right': [],
            'terrain_grids': [],
            'obstacle_grids': [],
            'states': [],     # [pos+orient]
            'velocities': [], # [linear+angular]
            'actions': [],    # [linear+angular]
            'locomotion_modes': [],
            'timestamps': []
        }

    def process(self):
        print(f"Reading bag: {self.bag_path}")
        storage_options, converter_options = get_rosbag_options(self.bag_path)
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topics = reader.get_all_topics_and_types()
        topic_types = {t.name: t.type for t in topics}

        # We will trigger a "step" based on the frontleft camera (approx 10Hz/5Hz)
        # But we enforce target_freq to avoid duplicate frames
        last_step_time = 0.0
        step_interval = 1.0 / self.target_freq
        
        count = 0
        
        while reader.has_next():
            (topic, data, t_ns) = reader.read_next()
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            
            t_sec = t_ns / 1e9
            
            # --- Update latest buffers ---
            if topic == '/odometry':
                self.latest[topic] = self._process_odom(msg)
            elif topic == '/cmd_vel':
                self.latest[topic] = self._process_cmd(msg)
            elif topic == '/status/mobility_params':
                self.latest[topic] = self._process_mobility_params(msg)
            elif 'image' in topic or 'grid' in topic:
                try:
                    encoding = "passthrough"
                    if "image" in topic and "depth" not in topic:
                        encoding = "rgb8"
                    elif "grid" in topic:
                        encoding = "32FC1"
                    
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
                    
                    # Rotate cameras if needed (Spot cams are often rotated 90 deg)
                    if topic in ['/camera/frontleft/image', '/camera/frontright/image']:
                        try:
                            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
                        except:
                            pass
                            
                    self.latest[topic] = cv_img
                    
                    # --- Trigger Step ---
                    # We use frontleft camera as the "clock" for the episode
                    if topic == '/camera/frontleft/image':
                        if t_sec - last_step_time >= step_interval:
                            self._save_step(t_sec)
                            last_step_time = t_sec
                            count += 1
                            if count % 50 == 0:
                                print(f"Processed {count} steps...", end='\r')
                                
                except Exception as e:
                    print(f"Error processing {topic}: {e}")

        print(f"\nFinished bag. Total steps: {count}")
        self._write_hdf5()

    def _process_odom(self, msg):
        # Extract State and Velocity
        pos = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        
        lin = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        ang = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        
        return {
            'state': np.array(pos + quat, dtype=np.float32),
            'velocity': np.array(lin + ang, dtype=np.float32)
        }

    def _process_cmd(self, msg):
        return np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ], dtype=np.float32)

    def _process_mobility_params(self, msg):
        # locomotion_hint is the numeric mode
        return int(msg.locomotion_hint)

    def _save_step(self, timestamp):
        # Perform "Zero-order hold" - take the latest available value for everything
        
        # 1. Images (Required)
        if self.latest['/camera/frontleft/image'] is None:
            return # Skip if we don't have the main camera yet
            
        self.episode['images_left'].append(self.latest['/camera/frontleft/image'])
        self.episode['timestamps'].append(timestamp)
        
        # 2. Other Images (Optional - fill with zeros or dupes if missing to keep shape)
        if self.latest['/camera/frontright/image'] is not None:
             self.episode['images_right'].append(self.latest['/camera/frontright/image'])
        else:
             self.episode['images_right'].append(np.zeros_like(self.latest['/camera/frontleft/image']))
             
        if self.latest['/camera/frontmiddle_virtual/image'] is not None:
            self.episode['images_front'].append(self.latest['/camera/frontmiddle_virtual/image'])
            
        # 3. Depth (Optional)
        if self.latest['/depth/frontleft/image'] is not None:
            self.episode['depth_left'].append(self.latest['/depth/frontleft/image'])
        if self.latest['/depth/frontright/image'] is not None:
            self.episode['depth_right'].append(self.latest['/depth/frontright/image'])

        # 4. Grids
        if self.latest['/spot/local_grid/terrain'] is not None:
             self.episode['terrain_grids'].append(self.latest['/spot/local_grid/terrain'])
        if self.latest['/spot/local_grid/obstacle_distance'] is not None:
             self.episode['obstacle_grids'].append(self.latest['/spot/local_grid/obstacle_distance'])

        # 5. Odom
        if self.latest['/odometry'] is not None:
            self.episode['states'].append(self.latest['/odometry']['state'])
            self.episode['velocities'].append(self.latest['/odometry']['velocity'])
        else:
            self.episode['states'].append(np.zeros(7, dtype=np.float32))
            self.episode['velocities'].append(np.zeros(6, dtype=np.float32))

        # 6. Action
        if self.latest['/cmd_vel'] is not None:
            self.episode['actions'].append(self.latest['/cmd_vel'])
        else:
            self.episode['actions'].append(np.zeros(6, dtype=np.float32))

        # 7. Locomotion mode
        if self.latest['/status/mobility_params'] is not None:
            self.episode['locomotion_modes'].append(self.latest['/status/mobility_params'])
        else:
            self.episode['locomotion_modes'].append(0)

    def _write_hdf5(self):
        if not self.episode['timestamps']:
            print("No data collected!")
            return

        bag_name = os.path.basename(self.bag_path.rstrip('/'))
        out_name = os.path.join(self.output_dir, f"{bag_name}_converted.h5")
        
        print(f"Saving to {out_name}...")
        
        with h5py.File(out_name, 'w') as f:
            # Compress images
            f.create_dataset('observations/images_left', data=np.stack(self.episode['images_left']), compression='gzip')
            f.create_dataset('observations/images_right', data=np.stack(self.episode['images_right']), compression='gzip')
            
            if self.episode['images_front']:
                f.create_dataset('observations/images_front', data=np.stack(self.episode['images_front']), compression='gzip')
            
            # Depth (usually uint16 or float32 - compress well)
            if self.episode['depth_left']:
                f.create_dataset('observations/depth_left', data=np.stack(self.episode['depth_left']), compression='gzip')
            if self.episode['depth_right']:
                 f.create_dataset('observations/depth_right', data=np.stack(self.episode['depth_right']), compression='gzip')

            # Grids
            if self.episode['terrain_grids']:
                f.create_dataset('observations/terrain_grids', data=np.stack(self.episode['terrain_grids']), compression='gzip')
            if self.episode['obstacle_grids']:
                f.create_dataset('observations/obstacle_grids', data=np.stack(self.episode['obstacle_grids']), compression='gzip')
            
            # Check for grid mismatch
            n_obs = len(self.episode['timestamps'])
            if len(self.episode['terrain_grids']) > 0 and len(self.episode['terrain_grids']) != n_obs:
                 print(f"Warning: Grid count mismatch ({len(self.episode['terrain_grids'])} vs {n_obs})")

            # Scalars/Vectors
            f.create_dataset('observations/state', data=np.array(self.episode['states']))
            f.create_dataset('observations/velocities', data=np.array(self.episode['velocities']))
            f.create_dataset('actions', data=np.array(self.episode['actions']))
            f.create_dataset('observations/locomotion_mode', data=np.array(self.episode['locomotion_modes'], dtype=np.uint32))
            f.create_dataset('timestamps', data=np.array(self.episode['timestamps']))
            
        print("Conversion Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert rosbag to HDF5")
    parser.add_argument('bag_path', nargs='?', default=None, help="Path to rosbag folder (optional)")
    parser.add_argument('--output', default=None, help="Output directory")
    parser.add_argument('--recorded-dir', default=DEFAULT_RECORD_DIR, help="Directory containing recorded bags")
    args = parser.parse_args()

    bag_path = args.bag_path
    if bag_path is None:
        recorded_dir = args.recorded_dir
        if not os.path.isdir(recorded_dir):
            print(f"Recorded directory not found: {recorded_dir}")
            sys.exit(1)
        # Pick the most recent bag directory
        candidates = [
            os.path.join(recorded_dir, d)
            for d in os.listdir(recorded_dir)
            if os.path.isdir(os.path.join(recorded_dir, d))
        ]
        if not candidates:
            print(f"No bag directories found in: {recorded_dir}")
            sys.exit(1)
        bag_path = max(candidates, key=os.path.getmtime)
        print(f"Auto-selected latest bag: {bag_path}")

    output_dir = args.output
    if output_dir is None:
        output_dir = bag_path.rstrip('/')

    converter = BagConverter(bag_path, output_dir)
    converter.process()
