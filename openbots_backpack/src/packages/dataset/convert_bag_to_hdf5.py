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
        self.target_freq = target_freq  # Default: 5 Hz (limited by camera hardware on SPOT)
        self.bridge = CvBridge()
        
        # Buffer for latest messages
        self.latest = {
            '/camera/frontmiddle_virtual/image': None,
            '/depth_registered/frontleft/image': None,
            '/depth_registered/frontright/image': None,
            '/spot/local_grid/terrain': None,
            '/spot/local_grid/obstacle_distance': None,
            '/odometry': None,
            '/cmd_vel': None,
            '/status/mobility_params': None
        }
        
        # Episode storage
        self.episode = {
            'images_front': [],           # Stitched front camera
            'depth_registered_left': [],  # Left depth
            'depth_registered_right': [], # Right depth
            'terrain_grids': [],          # Terrain height grids
            'states': [],                 # [pos+orient]
            'velocities': [],             # [linear+angular]
            'actions': [],                # [linear+angular]
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

        # We will trigger a "step" based on the frontmiddle camera (approx 10Hz/5Hz)
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
                    self.latest[topic] = cv_img
                    
                    # --- Trigger Step ---
                    # We use frontmiddle camera as the "clock" for the episode
                    if topic == '/camera/frontmiddle_virtual/image':
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
        
        # 1. Images (Required) - frontmiddle_virtual for stitched camera
        if self.latest['/camera/frontmiddle_virtual/image'] is None:
            return # Skip if we don't have the image yet
            
        self.episode['images_front'].append(self.latest['/camera/frontmiddle_virtual/image'])
        
        # 2. Registered depth maps
        if self.latest['/depth_registered/frontleft/image'] is not None:
            self.episode['depth_registered_left'].append(self.latest['/depth_registered/frontleft/image'])
        else:
            # Placeholder if missing
            self.episode['depth_registered_left'].append(np.zeros((64, 64), dtype=np.uint8))
        
        if self.latest['/depth_registered/frontright/image'] is not None:
            self.episode['depth_registered_right'].append(self.latest['/depth_registered/frontright/image'])
        else:
            # Placeholder if missing
            self.episode['depth_registered_right'].append(np.zeros((64, 64), dtype=np.uint8))
        
        # 3. Terrain grid (if available)
        if self.latest['/spot/local_grid/terrain'] is not None:
            self.episode['terrain_grids'].append(self.latest['/spot/local_grid/terrain'])
        
        self.episode['timestamps'].append(timestamp)

        # 2. Odom
        if self.latest['/odometry'] is not None:
            self.episode['states'].append(self.latest['/odometry']['state'])
            self.episode['velocities'].append(self.latest['/odometry']['velocity'])
        else:
            self.episode['states'].append(np.zeros(7, dtype=np.float32))
            self.episode['velocities'].append(np.zeros(6, dtype=np.float32))

        # 3. Action
        if self.latest['/cmd_vel'] is not None:
            self.episode['actions'].append(self.latest['/cmd_vel'])
        else:
            self.episode['actions'].append(np.zeros(6, dtype=np.float32))

        # 4. Locomotion mode
        if self.latest['/status/mobility_params'] is not None:
            self.episode['locomotion_modes'].append(self.latest['/status/mobility_params'])
        else:
            self.episode['locomotion_modes'].append(0)

    def _filter_mismatched_shapes(self):
        """Remove frames where sensor data has inconsistent shapes."""
        n_frames = len(self.episode['timestamps'])
        
        # Ensure all lists are the same length (pad or trim as needed)
        min_len = min(
            len(self.episode['images_front']),
            len(self.episode['depth_registered_left']),
            len(self.episode['depth_registered_right']),
            len(self.episode['terrain_grids']),
            len(self.episode['states']),
            len(self.episode['velocities']),
            len(self.episode['actions']),
            len(self.episode['locomotion_modes']),
            len(self.episode['timestamps'])
        )
        
        # Trim all to same length
        for key in self.episode:
            if isinstance(self.episode[key], list):
                self.episode[key] = self.episode[key][:min_len]
        
        print(f"Aligned all arrays to length {min_len}")
        
        # Now filter for left != right mismatch
        valid_indices = []
        for i in range(min_len):
            left_shape = self.episode['depth_registered_left'][i].shape
            right_shape = self.episode['depth_registered_right'][i].shape
            if left_shape == right_shape:
                valid_indices.append(i)
            else:
                print(f"  Frame {i}: depth mismatch (L:{left_shape} vs R:{right_shape})")
        
        if len(valid_indices) < min_len:
            print(f"Pass 1 filtering: {min_len} → {len(valid_indices)} frames (removed L!=R mismatches)")
            for key in self.episode:
                if isinstance(self.episode[key], list):
                    self.episode[key] = [self.episode[key][i] for i in valid_indices]
            min_len = len(valid_indices)
        
        # Now filter for consistent depth shape
        if min_len > 0:
            depth_shapes = [self.episode['depth_registered_left'][i].shape for i in range(min_len)]
            
            from collections import Counter
            shape_counts = Counter(depth_shapes)
            most_common_shape = shape_counts.most_common(1)[0][0]
            
            valid_indices = [i for i in range(min_len) if self.episode['depth_registered_left'][i].shape == most_common_shape]
            
            if len(valid_indices) < min_len:
                print(f"Pass 2 filtering: {min_len} → {len(valid_indices)} frames (kept shape {most_common_shape})")
                for key in self.episode:
                    if isinstance(self.episode[key], list):
                        self.episode[key] = [self.episode[key][i] for i in valid_indices]

    def _write_hdf5(self):
        if not self.episode['timestamps']:
            print("No data collected!")
            return

        # Filter out frames with mismatched shapes (keeps episode consistent)
        self._filter_mismatched_shapes()
        
        if not self.episode['timestamps']:
            print("No valid frames after filtering!")
            return

        bag_name = os.path.basename(self.bag_path.rstrip('/'))
        out_name = os.path.join(self.output_dir, f"{bag_name}_converted.h5")
        
        print(f"Saving to {out_name}...")
        
        with h5py.File(out_name, 'w') as f:
            # Save stitched front camera image
            if self.episode['images_front']:
                f.create_dataset('observations/images_front', data=np.stack(self.episode['images_front']), compression='gzip')
                print(f"  Saved {len(self.episode['images_front'])} stitched front images")
            
            # Save registered depth maps (stereo pair)
            if self.episode['depth_registered_left']:
                f.create_dataset('observations/depth_registered_left', data=np.stack(self.episode['depth_registered_left']), compression='gzip')
                print(f"  Saved {len(self.episode['depth_registered_left'])} left depth frames")
            
            if self.episode['depth_registered_right']:
                f.create_dataset('observations/depth_registered_right', data=np.stack(self.episode['depth_registered_right']), compression='gzip')
                print(f"  Saved {len(self.episode['depth_registered_right'])} right depth frames")
            
            # Save terrain grids if available
            if self.episode['terrain_grids']:
                f.create_dataset('observations/terrain_grids', data=np.stack(self.episode['terrain_grids']), compression='gzip')
                f.attrs['terrain_grid_cell_size'] = 0.03  # meters per cell
                print(f"  Saved {len(self.episode['terrain_grids'])} terrain height grids")
            
            # Scalars/Vectors
            f.create_dataset('observations/state', data=np.array(self.episode['states']))
            f.create_dataset('observations/velocities', data=np.array(self.episode['velocities']))
            f.create_dataset('actions', data=np.array(self.episode['actions']))
            f.create_dataset('observations/locomotion_mode', data=np.array(self.episode['locomotion_modes'], dtype=np.uint32))
            f.create_dataset('timestamps', data=np.array(self.episode['timestamps']))
            
            print(f"  Saved {len(self.episode['timestamps'])} steps total")
            
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
