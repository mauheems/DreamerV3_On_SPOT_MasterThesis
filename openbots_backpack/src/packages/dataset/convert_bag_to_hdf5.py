#!/usr/bin/env python3
"""
Convert SPOT Rosbags to DreamerV3-compatible HDF5 episodes.
Uses 'cv_bridge' and 'rclpy' serialization tools to parse bag data offline.
"""

import argparse
import os
import sys
import sqlite3
import numpy as np
import h5py
import cv2
from PIL import Image as PILImage
from scipy.ndimage import zoom
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge

# Message types import (adjust if needed for custom msgs)
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

DEFAULT_RECORD_DIR = "/home/ob/openbots_ws/src/packages/dataset/recorded_data"

def get_db3_file(bag_path):
    """Find the .db3 file in the bag directory. Returns None if no .db3 file found."""
    if os.path.isfile(bag_path) and bag_path.endswith('.db3'):
        return bag_path
    
    # Search for .db3 file in directory
    try:
        for f in os.listdir(bag_path):
            if f.endswith('.db3'):
                return os.path.join(bag_path, f)
    except (PermissionError, OSError):
        return None
    
    return None

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
            '/status/mobility_params': None,
            '/collision_event': False  # Boolean flag for collision events
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
            'collision_events': [],       # [bool] collision flags per timestep
            'timestamps': []
        }

    def process(self):
        print(f"Reading bag: {self.bag_path}")
        db3_file = get_db3_file(self.bag_path)
        print(f"Using database: {db3_file}")
        
        # Connect to SQLite database
        conn = sqlite3.connect(db3_file)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query topics table
        cursor.execute("SELECT id, name, type FROM topics")
        topics = {row['id']: {'name': row['name'], 'type': row['type']} for row in cursor.fetchall()}
        
        if not topics:
            print("No topics found in database!")
            conn.close()
            return
        
        print(f"Found {len(topics)} topics")
        
        # Query messages ordered by timestamp
        cursor.execute("SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp")
        messages = cursor.fetchall()
        
        if not messages:
            print("No messages found in database!")
            conn.close()
            return
        
        print(f"Found {len(messages)} messages")
        
        # Process messages
        last_step_time = 0.0
        step_interval = 1.0 / self.target_freq
        count = 0
        skipped_types = set()
        
        for msg_row in messages:
            topic_id = msg_row['topic_id']
            t_ns = msg_row['timestamp']
            data = msg_row['data']
            
            t_sec = t_ns / 1e9
            topic_info = topics[topic_id]
            topic_name = topic_info['name']
            msg_type_str = topic_info['type']
            
            # Try to deserialize the message type
            try:
                msg_type = get_message(msg_type_str)
                msg = deserialize_message(data, msg_type)
            except (ModuleNotFoundError, ImportError) as e:
                # Skip message types that aren't available (e.g., custom spot_msgs)
                if msg_type_str not in skipped_types:
                    skipped_types.add(msg_type_str)
                    print(f"  Skipping unknown message type: {msg_type_str}")
                continue
            
            # --- Update latest buffers ---
            if topic_name == '/odometry':
                self.latest[topic_name] = self._process_odom(msg)
            elif topic_name == '/cmd_vel':
                self.latest[topic_name] = self._process_cmd(msg)
            elif topic_name == '/status/mobility_params':
                self.latest[topic_name] = self._process_mobility_params(msg)
            elif topic_name == '/collision_event':
                # Capture collision flag (Bool message)
                self.latest[topic_name] = msg.data
            elif 'image' in topic_name or 'grid' in topic_name:
                try:
                    encoding = "passthrough"
                    if "image" in topic_name and "depth" not in topic_name:
                        encoding = "rgb8"
                    elif "grid" in topic_name:
                        encoding = "32FC1"
                    
                    cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
                    self.latest[topic_name] = cv_img
                    
                    # --- Trigger Step ---
                    # We use frontmiddle camera as the "clock" for the episode
                    if topic_name == '/camera/frontmiddle_virtual/image':
                        if t_sec - last_step_time >= step_interval:
                            self._save_step(t_sec)
                            last_step_time = t_sec
                            count += 1
                            if count % 50 == 0:
                                print(f"Processed {count} steps...", end='\r')
                                
                except Exception as e:
                    print(f"Error processing {topic_name}: {e}")

        print(f"\nFinished bag. Total steps: {count}")
        conn.close()
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
        # Return only first 4 dimensions: [linear.x, linear.y, linear.z, gait_selection]
        # Truncate from 6D to 4D to match environment action space
        return np.array([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x  # Only first angular component (gait selection)
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

        # 3. Action (4D: [vx, vy, vz, wz])
        if self.latest['/cmd_vel'] is not None:
            self.episode['actions'].append(self.latest['/cmd_vel'])
        else:
            self.episode['actions'].append(np.zeros(4, dtype=np.float32))

        # 4. Locomotion mode
        if self.latest['/status/mobility_params'] is not None:
            self.episode['locomotion_modes'].append(self.latest['/status/mobility_params'])
        else:
            self.episode['locomotion_modes'].append(0)
        
        # 5. Collision event flag
        self.episode['collision_events'].append(bool(self.latest['/collision_event']))
        
        # Reset collision flag after recording it
        self.latest['/collision_event'] = False

    def _normalize_locomotion_modes(self):
        """
        Normalize locomotion modes to binary classification:
        - 0: CRAWL (modes 4, 10)
        - 1: TROT variants (modes 1, 2, 3, 6) - includes AUTO, TROT, SPEED_SELECT_TROT, SPEED_SELECT_AMBLE
        """
        normalized = []
        mode_mapping = {
            4: 0,   # HINT_CRAWL
            10: 0,  # HINT_SPEED_SELECT_CRAWL
            1: 1,   # HINT_AUTO (converges to trot)
            2: 1,   # HINT_TROT
            3: 1,   # HINT_SPEED_SELECT_TROT
            6: 1,   # HINT_SPEED_SELECT_AMBLE
        }
        
        for mode in self.episode['locomotion_modes']:
            normalized.append(mode_mapping.get(mode, 1))  # Default to trot for unknown modes
        
        self.episode['locomotion_modes'] = normalized

    def _normalize_images(self):
        """
        Normalize images to [0, 1] range.
        RGB images: divide by 255
        Depth maps: already small values, but normalize by max depth (assume max ~10m)
        """
        # Normalize front RGB images
        if self.episode['images_front']:
            images_front_normalized = []
            for img in self.episode['images_front']:
                if img.dtype == np.uint8:
                    img_norm = img.astype(np.float32) / 255.0
                else:
                    img_norm = img.astype(np.float32)
                images_front_normalized.append(img_norm)
            self.episode['images_front'] = images_front_normalized
        
        # Normalize depth maps
        if self.episode['depth_registered_left']:
            depth_left_normalized = []
            for depth in self.episode['depth_registered_left']:
                if depth.dtype == np.uint8:
                    depth_norm = depth.astype(np.float32) / 255.0
                elif depth.dtype == np.uint16:
                    # Assuming 16-bit depth in millimeters, convert to meters then normalize
                    depth_norm = (depth.astype(np.float32) / 1000.0) / 10.0  # normalize by 10m
                else:
                    depth_norm = depth.astype(np.float32)
                depth_left_normalized.append(depth_norm)
            self.episode['depth_registered_left'] = depth_left_normalized
        
        if self.episode['depth_registered_right']:
            depth_right_normalized = []
            for depth in self.episode['depth_registered_right']:
                if depth.dtype == np.uint8:
                    depth_norm = depth.astype(np.float32) / 255.0
                elif depth.dtype == np.uint16:
                    depth_norm = (depth.astype(np.float32) / 1000.0) / 10.0
                else:
                    depth_norm = depth.astype(np.float32)
                depth_right_normalized.append(depth_norm)
            self.episode['depth_registered_right'] = depth_right_normalized

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
            len(self.episode['collision_events']),
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
        
        # Normalize images to [0, 1] range
        self._normalize_images()
        
        # Normalize locomotion modes to binary classification
        self._normalize_locomotion_modes()
        
        if not self.episode['timestamps']:
            print("No valid frames after filtering!")
            return

        bag_name = os.path.basename(self.bag_path.rstrip('/'))
        out_name = os.path.join(self.output_dir, f"{bag_name}_converted.h5")
        
        print(f"Saving to {out_name}...")
        
        # Retry logic for file locking issues on external drives
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                with h5py.File(out_name, 'w') as f:
                    # Save stitched front camera image, resized to (128, 49, 3)
                    if self.episode['images_front']:
                        resized_images = []
                        for img in self.episode['images_front']:
                            # Skip invalid shapes
                            if img.shape != (1662, 640, 3):
                                print(f"Skipping invalid image shape: {img.shape}")
                                continue
                            # Convert float images to uint8
                            if img.dtype != np.uint8:
                                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                            pil_img = PILImage.fromarray(img)
                            # Rotate 90 degrees clockwise to get panorama (wider than high)
                            pil_img = pil_img.rotate(-90, expand=True)
                            resized = pil_img.resize((128, 49), PILImage.BILINEAR)
                            resized_images.append(np.array(resized, dtype=np.uint8))
                        if resized_images:
                            f.create_dataset('observations/images_front', data=np.stack(resized_images), compression='gzip')
                        f.attrs['images_front_resized'] = True
                        f.attrs['images_front_shape'] = '(128, 49, 3)'
                        print(f"  Saved {len(resized_images)} stitched front images (resized to 128x49)")

                    # Save registered depth maps (stereo pair)
                    if self.episode['depth_registered_left']:
                        f.create_dataset('observations/depth_registered_left', data=np.stack(self.episode['depth_registered_left']), compression='gzip')
                        f.attrs['depth_registered_left_normalized'] = True
                        f.attrs['depth_registered_left_range'] = '[0, 1] (normalized by 10m max)'
                        print(f"  Saved {len(self.episode['depth_registered_left'])} left depth frames (normalized)")

                    if self.episode['depth_registered_right']:
                        f.create_dataset('observations/depth_registered_right', data=np.stack(self.episode['depth_registered_right']), compression='gzip')
                        f.attrs['depth_registered_right_normalized'] = True
                        f.attrs['depth_registered_right_range'] = '[0, 1] (normalized by 10m max)'
                        print(f"  Saved {len(self.episode['depth_registered_right'])} right depth frames (normalized)")

                    # Save terrain grids if available (normalize per-frame: ground level = 0, then resize to 64x64)
                    if self.episode['terrain_grids']:
                        terrain_grids_normalized = []
                        for grid in self.episode['terrain_grids']:
                            grid_copy = grid.copy() if isinstance(grid, np.ndarray) else np.array(grid)
                            min_height = np.nanmin(grid_copy)
                            if not np.isnan(min_height):
                                grid_copy = grid_copy - min_height
                            # Resize to (64, 64)
                            zoom_factor = 64 / grid_copy.shape[0]
                            resized_grid = zoom(grid_copy, zoom_factor, order=1).astype(np.float32)
                            terrain_grids_normalized.append(resized_grid)
                        f.create_dataset('observations/terrain_grids', data=np.stack(terrain_grids_normalized), compression='gzip')
                        f.attrs['terrain_grid_cell_size'] = 0.03  # meters per cell
                        f.attrs['terrain_grids_resized'] = True
                        f.attrs['terrain_grids_shape'] = '(64, 64)'
                        f.attrs['terrain_normalization_note'] = 'Per-frame normalization: ground level set to 0m, resized to 64x64'
                        print(f"  Saved {len(terrain_grids_normalized)} terrain height grids (normalized and resized to 64x64)")

                    # Scalars/Vectors
                    # Save states and rotate velocities from world -> body frame using per-sample yaw
                    states_arr = np.array(self.episode['states'])
                    vel_arr = np.array(self.episode['velocities'])

                    # Write states as-is
                    f.create_dataset('observations/state', data=states_arr)

                    # If states contain quaternions (qx,qy,qz,qw) rotate linear vx,vy into body frame
                    try:
                        if states_arr.ndim == 2 and states_arr.shape[1] >= 7 and vel_arr.ndim == 2 and vel_arr.shape[0] > 0:
                            T = min(states_arr.shape[0], vel_arr.shape[0])
                            q = states_arr[:T, 3:7]
                            qx = q[:, 0]; qy = q[:, 1]; qz = q[:, 2]; qw = q[:, 3]
                            # yaw from quaternion
                            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

                            vwx_orig = vel_arr[:T, 0].astype(np.float32).copy()
                            vwy_orig = vel_arr[:T, 1].astype(np.float32).copy() if vel_arr.shape[1] > 1 else np.zeros_like(vwx_orig)

                            vbx = np.cos(yaw) * vwx_orig + np.sin(yaw) * vwy_orig
                            vby = -np.sin(yaw) * vwx_orig + np.cos(yaw) * vwy_orig

                            vel_rot = vel_arr.copy()
                            vel_rot[:T, 0] = vbx
                            if vel_rot.shape[1] > 1:
                                vel_rot[:T, 1] = vby

                            f.create_dataset('observations/velocities', data=vel_rot)

                            # provenance
                            f.attrs['velocities_frame'] = 'body'
                            f.attrs['velocities_rotated_to_body'] = True
                            f.attrs['velocities_rotated_by'] = 'convert_bag_to_hdf5.py'
                            try:
                                f.attrs['vx_mean_before'] = float(np.nanmean(vwx_orig))
                                f.attrs['vx_frac_negative_before'] = float(np.sum(vwx_orig < 0) / float(len(vwx_orig)))
                            except Exception:
                                pass
                        else:
                            # No quaternion available — write velocities unchanged
                            f.create_dataset('observations/velocities', data=vel_arr)
                            f.attrs['velocities_frame'] = 'world'
                            f.attrs['velocities_rotated_to_body'] = False
                    except Exception as e:
                        # On any error, fall back to writing original velocities
                        print(f"Warning: unable to rotate velocities for {out_name}: {e}")
                        f.create_dataset('observations/velocities', data=vel_arr)
                        f.attrs['velocities_frame'] = 'world'
                        f.attrs['velocities_rotated_to_body'] = False
                    f.create_dataset('actions', data=np.array(self.episode['actions']))
                    f.create_dataset('observations/locomotion_mode', data=np.array(self.episode['locomotion_modes'], dtype=np.uint32))
                    f.create_dataset('observations/collision_events', data=np.array(self.episode['collision_events'], dtype=bool))
                    f.create_dataset('timestamps', data=np.array(self.episode['timestamps']))

                    print(f"  Saved {len(self.episode['timestamps'])} steps total")
                    if any(self.episode['collision_events']):
                        n_collisions = sum(self.episode['collision_events'])
                        print(f"  Recorded {n_collisions} collision events during episode")
                
                print("Conversion Complete!")
                return  # Successfully written, exit retry loop
                
            except (BlockingIOError, OSError) as e:
                if attempt < max_retries - 1:
                    import time
                    print(f"  File lock error (attempt {attempt+1}/{max_retries}): {e}")
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"  Failed to write after {max_retries} attempts: {e}")
                    raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert rosbag to HDF5")
    parser.add_argument('bag_path', nargs='?', default=None, help="Path to rosbag folder (optional)")
    parser.add_argument('--output', default=None, help="Output directory")
    parser.add_argument('--recorded-dir', default=None, help="Directory containing recorded bags (overrides default)")
    parser.add_argument('--harddrive', type=str, default=None, help="Path to external harddrive recorded_data directory")
    parser.add_argument('--batch', action='store_true', help="Convert all bags in recorded directory")
    args = parser.parse_args()

    bag_path = args.bag_path
    
    # Determine which recorded_dir to use
    if args.harddrive:
        recorded_dir = args.harddrive
        print(f"Using external harddrive: {recorded_dir}")
    elif args.recorded_dir:
        recorded_dir = args.recorded_dir
    else:
        recorded_dir = DEFAULT_RECORD_DIR
        print(f"Using local repository: {recorded_dir}")
    
    # Batch mode: convert all bags in recorded directory
    if args.batch or bag_path is None:
        print(f"DEBUG: Checking if path is directory: {os.path.isdir(recorded_dir)}")
        print(f"DEBUG: Path exists: {os.path.exists(recorded_dir)}")
        print(f"DEBUG: Absolute path: {os.path.abspath(recorded_dir)}")
        
        if not os.path.isdir(recorded_dir):
            print(f"Recorded directory not found: {recorded_dir}")
            sys.exit(1)
        
        # Get all bag directories (filter for valid rosbags only)
        all_dirs = [
            os.path.join(recorded_dir, d)
            for d in sorted(os.listdir(recorded_dir))
            if os.path.isdir(os.path.join(recorded_dir, d))
        ]
        
        # Filter to only directories that contain .db3 files
        candidates = [d for d in all_dirs if get_db3_file(d) is not None]
        
        if not candidates:
            print(f"No bag directories found in: {recorded_dir}")
            sys.exit(1)
        
        # Set output directory to processed_data
        processed_data_dir = os.path.join(os.path.dirname(recorded_dir), 'processed_data')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        print(f"Found {len(candidates)} valid bag(s) to convert (skipped {len(all_dirs) - len(candidates)} non-bag directories):")
        for i, bag in enumerate(candidates, 1):
            print(f"  {i}. {os.path.basename(bag)}")
        print(f"Output directory: {processed_data_dir}\n")
        
        # Convert each bag (skip already converted)
        successful = 0
        failed = 0
        skipped = 0
        for i, bag_path in enumerate(candidates, 1):
            bag_name = os.path.basename(bag_path)
            output_file = os.path.join(processed_data_dir, f"{bag_name}_converted.h5")
            
            # Check if already converted
            if os.path.exists(output_file):
                print(f"\n{'='*60}")
                print(f"Skipping bag {i}/{len(candidates)}: {bag_name}")
                print(f"  (already converted to {os.path.basename(output_file)})")
                print(f"{'='*60}")
                skipped += 1
                continue
            
            print(f"\n{'='*60}")
            print(f"Converting bag {i}/{len(candidates)}: {bag_name}")
            print(f"{'='*60}")
            converter = BagConverter(bag_path, processed_data_dir)
            try:
                converter.process()
                successful += 1
            except Exception as e:
                print(f"ERROR processing {bag_name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                continue
        
        print(f"\n{'='*60}")
        print(f"Batch conversion complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Skipped (already converted): {skipped}")
        print(f"Output saved to: {processed_data_dir}")
        print(f"{'='*60}")
    else:
        # Single bag mode
        output_dir = args.output
        if output_dir is None:
            output_dir = bag_path.rstrip('/')

        converter = BagConverter(bag_path, output_dir)
        converter.process()
