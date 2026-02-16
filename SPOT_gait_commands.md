# SPOT gait & motion commands

## Commands used

- Set locomotion hint (gait):
  
	ros2 service call /locomotion_mode spot_msgs/srv/SetLocomotion "{locomotion_mode: 1}"

- Walk forward 0.5 m (body frame):
  
	ros2 action send_goal /trajectory spot_msgs/action/Trajectory "{target_pose: {header: {frame_id: 'body'}, pose: {position: {x: 0.5, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}, duration: {sec: 5}, precise_positioning: false}"

- Rotate in place (body frame, 90° left):
  
	ros2 action send_goal /trajectory spot_msgs/action/Trajectory "{target_pose: {header: {frame_id: 'body'}, pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {z: 0.7071, w: 0.7071}}}, duration: {sec: 5}, precise_positioning: false}"

- Stand:
  
	ros2 service call /stand std_srvs/srv/Trigger

- Sit:
  
	ros2 service call /sit std_srvs/srv/Trigger

- Teleop
	ros2 run teleop_twist_keyboard teleop_twist_keyboard

## LocomotionHint values (Boston Dynamics)

| Name | Number | Description |
| --- | --- | --- |
| HINT_UNKNOWN | 0 | Invalid; do not use. |
| HINT_AUTO | 1 | No hint, robot chooses an appropriate gait (typically trot.) |
| HINT_TROT | 2 | Most robust gait which moves diagonal legs together. |
| HINT_SPEED_SELECT_TROT | 3 | Trot which comes to a stand when not commanded to move. |
| HINT_CRAWL | 4 | Slow and steady gait which moves only one foot at a time. |
| HINT_AMBLE | 5 | Four beat gait where one foot touches down at a time. |
| HINT_SPEED_SELECT_AMBLE | 6 | Amble which comes to a stand when not commanded to move. |
| HINT_JOG | 7 | Demo gait which moves diagonal leg pairs together with an aerial phase. |
| HINT_HOP | 8 | Demo gait which hops while holding some feet in the air. |
| HINT_SPEED_SELECT_CRAWL | 10 | Crawl which comes to a stand when not commanded to move. |
| HINT_AUTO_TROT | 3 | Deprecated (use HINT_SPEED_SELECT_TROT). |
| HINT_AUTO_AMBLE | 6 | Deprecated (use HINT_SPEED_SELECT_AMBLE). |