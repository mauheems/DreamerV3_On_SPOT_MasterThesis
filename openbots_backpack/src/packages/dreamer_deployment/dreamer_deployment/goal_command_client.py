#!/usr/bin/env python3
"""
Send a goal waypoint to the running Dreamer policy node.

Usage (after `colcon build && source install/setup.bash`):
    ros2 run dreamer_deployment goal_command_client --x 2.0 --y 1.0
    ros2 run dreamer_deployment goal_command_client --x 2.0 --y 1.0 --robot spot

The goal is given in the world (odom) frame.  The policy receives the goal as a
relative 2D vector [goal_x - robot_x, goal_y - robot_y] so it's always in robot-relative
coordinates internally — you always provide absolute world-frame coordinates here.
"""

import argparse
import rclpy
from rclpy.client import Client
from rclpy.node import Node


class GoalCommandClient:
    """One-shot client: set a navigation goal then exit."""

    # The policy node always listens on this fixed topic regardless of robot_name
    # because the node itself is not namespaced.
    SERVICE_NAME = '/spot/policy/set_goal'

    def __init__(self):
        rclpy.init()
        self.node = Node('goal_command_client')

        try:
            from dreamer_deployment.srv import SetGoalWaypoint
            self._srv_type = SetGoalWaypoint
        except ImportError:
            self.node.get_logger().warn(
                'SetGoalWaypoint not found (package not built yet) — '
                'using std_srvs/Trigger as fallback.'
            )
            from std_srvs.srv import Trigger
            self._srv_type = Trigger

        self._client: Client = self.node.create_client(
            self._srv_type, self.SERVICE_NAME
        )

        timeout = 5.0
        if not self._client.wait_for_service(timeout_sec=timeout):
            self.node.get_logger().error(
                f'Service {self.SERVICE_NAME} not available after {timeout}s — '
                'is the policy node running?'
            )
            self.node.destroy_node()
            rclpy.shutdown()
            raise RuntimeError('Policy node not reachable.')

        self.node.get_logger().info(f'Connected to {self.SERVICE_NAME}')

    def send_goal(self, x: float, y: float):
        """Send goal (world frame) to the policy node."""
        try:
            req = self._srv_type.Request()
            if hasattr(req, 'x'):
                req.x = float(x)
                req.y = float(y)
                self.node.get_logger().info(f'Sending goal: x={x:.2f}  y={y:.2f}')
            else:
                self.node.get_logger().info('Triggering service (fallback mode).')

            future = self._client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future)

            resp = future.result()
            if getattr(resp, 'success', True):
                self.node.get_logger().info(
                    f'✓ {getattr(resp, "message", "Goal accepted")}'
                )
            else:
                self.node.get_logger().error(
                    f'✗ {getattr(resp, "message", "Goal rejected")}'
                )
            return resp

        except Exception as e:
            self.node.get_logger().error(f'Service call failed: {e}')
            return None
        finally:
            self.node.destroy_node()
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description='Send a navigation goal (world frame) to the Dreamer policy node.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--x', type=float, required=True,  help='Goal X in world frame (m)')
    parser.add_argument('--y', type=float, required=True,  help='Goal Y in world frame (m)')
    args = parser.parse_args()

    client = GoalCommandClient()
    client.send_goal(args.x, args.y)


if __name__ == '__main__':
    main()
