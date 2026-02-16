#!/usr/bin/env python3
"""
Interactive ROS 2 bag recorder for splitting long runs into episodes.

Controls:
    space : start/stop episode (toggle)
    q     : stop (if recording), save, and exit

Usage:
    ./episode_bag_recorder.py

Notes:
- This script spawns `ros2 bag record` as a subprocess and sends SIGINT to stop it.
"""

import os
import signal
import subprocess
import sys
import time
import tty
import termios
from datetime import datetime

OUTPUT_DIR = "/home/ob/openbots_ws/src/packages/dataset/recorded_data"
TOPICS = [
    "/camera/frontmiddle_virtual/image",
    "/depth_registered/frontleft/image",
    "/depth_registered/frontright/image",
    "/odometry",
    "/cmd_vel",
    "/status/mobility_params",
    "/spot/local_grid/terrain",
    "/spot/local_grid/obstacle_distance",
    "/tf",
    "/tf_static",
]


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def make_bag_name(mode: str, difficulty: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_mode = mode.strip().lower().replace(" ", "-") or "unknown"
    safe_difficulty = difficulty.strip().lower().replace(" ", "-") or "unknown"
    return f"spot_bag_{timestamp}_{safe_mode}_{safe_difficulty}"


def start_recording(output_dir: str, bag_name: str) -> subprocess.Popen:
    cmd = ["ros2", "bag", "record", "--output", os.path.join(output_dir, bag_name)] + TOPICS
    return subprocess.Popen(cmd)


def stop_recording(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mode = input("Mode (expert/explore/autowalk/...): ").strip()
    difficulty = input("Difficulty (easy/med/hard/...): ").strip()

    print("\nInteractive Episode Recorder")
    print("space: start/stop episode | q: stop & quit")
    print(f"Output dir: {OUTPUT_DIR}")

    recording = None
    try:
        while True:
            key = getch()
            if key == " ":
                if recording and recording.poll() is None:
                    print("\n■ Stopping episode...")
                    stop_recording(recording)
                else:
                    bag_name = make_bag_name(mode, difficulty)
                    print(f"\n▶ Starting episode: {bag_name}")
                    recording = start_recording(OUTPUT_DIR, bag_name)
            elif key.lower() == "q":
                if recording and recording.poll() is None:
                    print("\n■ Stopping and exiting...")
                    stop_recording(recording)
                break
    except KeyboardInterrupt:
        if recording and recording.poll() is None:
            stop_recording(recording)


if __name__ == "__main__":
    main()
