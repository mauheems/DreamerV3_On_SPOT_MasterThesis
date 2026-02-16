#!/usr/bin/env python3
"""
Test SPOT obstacle_distance local grid access via Boston Dynamics SDK
"""

import os
import time
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.api import local_grid_pb2
import numpy as np
import matplotlib.pyplot as plt

def get_data_type(grid_proto):
    """Get numpy data type based on cell format"""
    # Map available cell formats
    cell_format_map = {
        0: np.uint8,      # CELL_FORMAT_UNKNOWN
        1: np.uint8,      # CELL_FORMAT_UINT8
        2: np.int8,       # CELL_FORMAT_INT8
        3: np.uint16,     # CELL_FORMAT_UINT16
        4: np.int16,      # CELL_FORMAT_INT16
        5: np.uint32,     # CELL_FORMAT_UINT32 (if exists)
        6: np.int32,      # CELL_FORMAT_INT32 (if exists)
        7: np.float32,    # CELL_FORMAT_FLOAT32
        8: np.float64,    # CELL_FORMAT_FLOAT64
    }
    
    # Default to int16 if format not recognized
    return cell_format_map.get(grid_proto.cell_format, np.int16)

def unpack_grid(grid_proto):
    """Unpack grid data following Boston Dynamics SDK pattern"""
    expected_size = grid_proto.extent.num_cells_x * grid_proto.extent.num_cells_y
    data_bytes = len(grid_proto.data)
    rle_count_sum = sum(grid_proto.rle_counts) if grid_proto.rle_counts else 0
    
    # Key insight: if rle_counts sum to expected_size, data is RLE encoded
    # Even if encoding field says RAW, the presence of rle_counts indicates RLE
    if rle_count_sum == expected_size and len(grid_proto.rle_counts) > 0:
        # Determine data type from buffer size vs number of RLE entries
        num_rle_entries = len(grid_proto.rle_counts)
        bytes_per_entry = data_bytes / num_rle_entries
        # Figure out dtype based on bytes per entry
        if abs(bytes_per_entry - 1.0) < 0.1:
            data_type = np.int8
        elif abs(bytes_per_entry - 2) < 0.1:
            data_type = np.int16
        elif abs(bytes_per_entry - 4) < 0.1:
            data_type = np.float32
        else:
            data_type = np.int8  # default to int8 for ambiguous cases
        
        # Parse and expand RLE
        cells_data = np.frombuffer(grid_proto.data, dtype=data_type)
        full_grid = np.repeat(cells_data[:len(grid_proto.rle_counts)], grid_proto.rle_counts).astype(data_type)
        
    else:
        # Raw data - figure out dtype
        if data_bytes == expected_size * 2:
            data_type = np.int16
        elif data_bytes == expected_size * 4:
            data_type = np.float32
        else:
            data_type = np.int16
        
        full_grid = np.frombuffer(grid_proto.data, dtype=data_type)
    
    # Apply scale and offset
    full_grid_float = full_grid.astype(np.float64)
    if grid_proto.cell_value_scale != 0:
        full_grid_float *= grid_proto.cell_value_scale
    full_grid_float += grid_proto.cell_value_offset
    
    return full_grid_float

def test_obstacle_distance_grid():
    """Test obstacle_distance and terrain grid access"""
    
    # Connection settings
    hostname = os.environ.get('SPOT_IP', '192.168.10.102')
    username = os.environ.get('BOSDYN_CLIENT_USERNAME', 'user')
    password = os.environ.get('BOSDYN_CLIENT_PASSWORD', 'corspotuser1')
    
    print(f"🤖 Testing SPOT grids...")
    print(f"📡 Connecting to {hostname}...")
    
    try:
        # Create SDK client and authenticate
        sdk = bosdyn.client.create_standard_sdk('grid_test')
        robot = sdk.create_robot(hostname)
        robot.authenticate(username, password)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync()
        print("  ✅ Connected and authenticated")
        
        # Create local grid client
        print("\n🗺️ Fetching grids...")
        local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)
        
        # Get available grid types
        grid_types = local_grid_client.get_local_grid_types()
        print(f"  Available grid types: {[gt.name for gt in grid_types]}")
        
        # Fetch both grids
        local_grid_responses = local_grid_client.get_local_grids(['obstacle_distance', 'terrain', 'terrain_valid'])
        
        if not local_grid_responses:
            print("❌ No response received")
            return False
        
        success = True
        
        # Process each grid response
        for response in local_grid_responses:
            if response.local_grid_type_name == 'obstacle_distance':
                print(f"\n📍 Processing obstacle_distance grid...")
                success &= visualize_obstacle_grid(response)
            elif response.local_grid_type_name == 'terrain':
                print(f"\n📍 Processing terrain grid...")
                success &= visualize_terrain_grid(response)
            elif response.local_grid_type_name == 'terrain_valid':
                print(f"\n📍 Processing terrain_valid grid...")
                success &= visualize_terrain_valid_grid(response)
        
        return success
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_terrain_grid_loop(interval_seconds=0.1, save_png_every=5, save_npy=True):
    """Continuously fetch terrain grid at a fixed interval."""

    # Connection settings
    hostname = os.environ.get('SPOT_IP', '192.168.10.102')
    username = os.environ.get('BOSDYN_CLIENT_USERNAME', 'user')
    password = os.environ.get('BOSDYN_CLIENT_PASSWORD', 'corspotuser1')

    print("🤖 Testing SPOT terrain grid loop...")
    print(f"📡 Connecting to {hostname}...")

    try:
        # Create SDK client and authenticate
        sdk = bosdyn.client.create_standard_sdk('terrain_grid_loop')
        robot = sdk.create_robot(hostname)
        robot.authenticate(username, password)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync()
        print("  ✅ Connected and authenticated")

        # Create local grid client
        local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)

        output_dir = os.path.join(os.path.dirname(__file__), 'recorded_data', 'terrain_height_grid')
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n🗺️ Fetching terrain grid every {interval_seconds}s (Ctrl+C to stop)...")

        iteration = 0
        confirmed_save = False
        last_rate_check = time.perf_counter()
        rate_window = 50
        while True:
            iteration += 1
            loop_start = time.perf_counter()

            local_grid_responses = local_grid_client.get_local_grids(['terrain'])
            if not local_grid_responses:
                pass
            else:
                for response in local_grid_responses:
                    if response.local_grid_type_name == 'terrain':
                        save_png = save_png_every > 0 and (iteration % save_png_every == 0)
                        npy_path, png_path = visualize_terrain_grid(
                            response,
                            output_dir=output_dir,
                            save_png=save_png,
                            save_npy=save_npy,
                        )
                        if not confirmed_save:
                            saved_path = npy_path or png_path
                            if saved_path and os.path.exists(saved_path):
                                print(f"✅ Saved sample to: {saved_path}")
                                confirmed_save = True

            if iteration % rate_window == 0:
                now = time.perf_counter()
                elapsed = now - last_rate_check
                if elapsed > 0:
                    achieved_hz = rate_window / elapsed
                    print(f"📈 Achieved rate: {achieved_hz:.2f} Hz")
                last_rate_check = now

            loop_elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, interval_seconds - loop_elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Stopped terrain grid loop")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_obstacle_grid(response):
    """Visualize obstacle_distance grid"""
    try:
        # Check status
        status_names = {
            0: "STATUS_OK",
            1: "STATUS_UNKNOWN_GRID_TYPE", 
            2: "STATUS_NO_SUCH_GRID",
            3: "STATUS_DATA_UNAVAILABLE"
        }
        status_name = status_names.get(response.status, f"STATUS_{response.status}")
        print(f"  Status: {status_name}")
        
        # Check if we have grid data
        if not response.local_grid or not response.local_grid.data:
            print("❌ No grid data in response")
            return False
        
        grid = response.local_grid
        extent = grid.extent
        
        print(f"  ✅ Successfully fetched obstacle_distance grid!")
        print(f"    📏 Dimensions: {extent.num_cells_x} x {extent.num_cells_y} cells")
        print(f"    📐 Cell size: {extent.cell_size:.3f} meters")
        print(f"    🔍 Encoding: {grid.encoding} ({'RLE' if grid.encoding == 1 else 'RAW'})")
        print(f"    💾 Data: {len(grid.data)} bytes")
        
        # Decode grid data
        data_array = unpack_grid(grid)
        expected_size = extent.num_cells_x * extent.num_cells_y
        
        if len(data_array) != expected_size:
            print(f"    ⚠️  Size mismatch! Truncating/padding to fit...")
            if len(data_array) > expected_size:
                data_array = data_array[:expected_size]
            else:
                data_array = np.pad(data_array, (0, expected_size - len(data_array)))
        
        # Reshape to 2D grid
        grid_2d = data_array.reshape(extent.num_cells_y, extent.num_cells_x)
        print(f"    📈 Values: min={grid_2d.min():.1f}, max={grid_2d.max():.1f}, mean={grid_2d.mean():.1f}")
        
        # Calculate bounds (centered around robot)
        width = extent.num_cells_x * extent.cell_size
        height = extent.num_cells_y * extent.cell_size
        bounds = [-width/2, width/2, -height/2, height/2]
        print(f"    📍 Bounds: ({bounds[0]:.2f}, {bounds[2]:.2f}) to ({bounds[1]:.2f}, {bounds[3]:.2f})")
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(grid_2d, cmap='viridis', origin='lower', extent=bounds)
        plt.colorbar(label='Obstacle Distance (mm)')
        plt.title(f'SPOT Obstacle Distance Grid\n{extent.num_cells_x}x{extent.num_cells_y} cells @ {extent.cell_size:.3f}m resolution')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True, alpha=0.3)
        
        # Mark robot position
        plt.plot(0, 0, 'r*', markersize=20, label='Robot')
        plt.legend()
        
        output_file = '/tmp/spot_obstacle_distance_grid.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    📊 Visualization saved to: {output_file}")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_terrain_grid(response, output_dir, save_png=True, save_npy=True):
    """Save terrain (height) grid"""
    try:
        # Check if we have grid data
        if not response.local_grid or not response.local_grid.data:
            return None, None
        
        grid = response.local_grid
        extent = grid.extent
        
        # Decode grid data
        data_array = unpack_grid(grid)
        expected_size = extent.num_cells_x * extent.num_cells_y
        
        if len(data_array) != expected_size:
            if len(data_array) > expected_size:
                data_array = data_array[:expected_size]
            else:
                data_array = np.pad(data_array, (0, expected_size - len(data_array)))
        
        # Reshape to 2D grid
        grid_2d = data_array.reshape(extent.num_cells_y, extent.num_cells_x)

        now = time.time()
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(now))
        millis = int((now - int(now)) * 1000)
        base_name = f'spot_terrain_grid_{timestamp}_{millis:03d}'

        npy_path = None
        png_path = None

        if save_npy:
            npy_path = os.path.join(output_dir, f'{base_name}.npy')
            np.save(npy_path, grid_2d)

        if save_png:
            # Calculate bounds (centered around robot)
            width = extent.num_cells_x * extent.cell_size
            height = extent.num_cells_y * extent.cell_size
            bounds = [-width/2, width/2, -height/2, height/2]

            plt.figure(figsize=(8, 6))
            plt.imshow(grid_2d, cmap='terrain', origin='lower', extent=bounds)
            plt.colorbar(label='Terrain Height (meters)')
            plt.title(f'SPOT Terrain Height Grid\n{extent.num_cells_x}x{extent.num_cells_y} cells @ {extent.cell_size:.3f}m resolution')
            plt.xlabel('X (meters)')
            plt.ylabel('Y (meters)')
            plt.grid(True, alpha=0.2)
            plt.plot(0, 0, 'r*', markersize=12, label='Robot')
            plt.legend()

            png_path = os.path.join(output_dir, f'{base_name}.png')
            plt.savefig(png_path, dpi=120, bbox_inches='tight')
            plt.close()
        
        return npy_path, png_path
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_terrain_valid_grid(response):
    """Visualize terrain_valid grid (validity mask)"""
    try:
        # Check status
        status_names = {
            0: "STATUS_OK",
            1: "STATUS_UNKNOWN_GRID_TYPE", 
            2: "STATUS_NO_SUCH_GRID",
            3: "STATUS_DATA_UNAVAILABLE"
        }
        status_name = status_names.get(response.status, f"STATUS_{response.status}")
        print(f"  Status: {status_name}")
        
        # Check if we have grid data
        if not response.local_grid or not response.local_grid.data:
            print("❌ No grid data in response")
            return False
        
        grid = response.local_grid
        extent = grid.extent
        
        print(f"  ✅ Successfully fetched terrain_valid grid!")
        print(f"    📏 Dimensions: {extent.num_cells_x} x {extent.num_cells_y} cells")
        print(f"    📐 Cell size: {extent.cell_size:.3f} meters")
        print(f"    🔍 Encoding: {grid.encoding} ({'RLE' if grid.encoding == 1 else 'RAW'})")
        print(f"    💾 Data: {len(grid.data)} bytes")
        
        # Decode grid data
        data_array = unpack_grid(grid)
        expected_size = extent.num_cells_x * extent.num_cells_y
        
        if len(data_array) != expected_size:
            print(f"    ⚠️  Size mismatch! Truncating/padding to fit...")
            if len(data_array) > expected_size:
                data_array = data_array[:expected_size]
            else:
                data_array = np.pad(data_array, (0, expected_size - len(data_array)))
        
        # Reshape to 2D grid
        grid_2d = data_array.reshape(extent.num_cells_y, extent.num_cells_x)
        print(f"    📊 Validity values: min={grid_2d.min():.1f}, max={grid_2d.max():.1f}, mean={grid_2d.mean():.1f}")
        
        # Count valid cells
        valid_count = np.sum(grid_2d > 0)
        total_count = grid_2d.size
        print(f"    ✓ Valid cells: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
        
        # Calculate bounds (centered around robot)
        width = extent.num_cells_x * extent.cell_size
        height = extent.num_cells_y * extent.cell_size
        bounds = [-width/2, width/2, -height/2, height/2]
        print(f"    📍 Bounds: ({bounds[0]:.2f}, {bounds[2]:.2f}) to ({bounds[1]:.2f}, {bounds[3]:.2f})")
        
        # Create visualization - use binary colormap for validity
        plt.figure(figsize=(12, 10))
        plt.imshow(grid_2d, cmap='RdYlGn', origin='lower', extent=bounds, vmin=0, vmax=1)
        plt.colorbar(label='Terrain Validity (0=invalid, 1=valid)')
        plt.title(f'SPOT Terrain Validity Grid\n{extent.num_cells_x}x{extent.num_cells_y} cells @ {extent.cell_size:.3f}m resolution')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True, alpha=0.3)
        
        # Mark robot position
        plt.plot(0, 0, 'b*', markersize=20, label='Robot')
        plt.legend()
        
        output_file = '/tmp/spot_terrain_valid_grid.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    📊 Visualization saved to: {output_file}")
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SPOT Grids Test")
    print("=" * 60)

    success = test_terrain_grid_loop(interval_seconds=0.1, save_png_every=5, save_npy=True)
    
    print("=" * 60)
    if success:
        print("🎉 Grids are ACCESSIBLE!")
        print("💡 Ready for integration into ROS2 bridge")
    else:
        print("😔 Could not access grids")
        print("💡 Make sure robot is powered on and standing")
    print("=" * 60)