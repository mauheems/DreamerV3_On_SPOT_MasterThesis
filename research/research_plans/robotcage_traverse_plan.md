# 🦾 **Dreamer-4 Model-Based RL for Obstacle-Rich Quadruped Navigation (Using ROS 2 + Spot SDK)**

### **Master Thesis Project Plan (Unified Markdown Version)**

---

# **1. High-Level Overview**

This thesis explores **model-based reinforcement learning** using **Dreamer-4** to navigate a Boston Dynamics **Spot** quadruped robot through a **caged obstacle environment**.
The robot learns from a large **teleoperated dataset**, using a world model to perform **latent imagination rollouts**, eventually producing actions that generalize to **new obstacle configurations**.

Spot is controlled using a university-provided **ROS 2 framework** (from the `openbots_backpack` repo) that wraps the official Boston Dynamics **Spot SDK**, adds networking, localization, and exposes a clean set of ROS topics and services for research.

---

# **2. System Architecture**

## **2.1 Hardware**

* Boston Dynamics Spot robot
* Raspberry Pi 5 mounted on Spot
* Teltonika RUTC50 router (Spot ⇆ RPi ⇆ workstation networking)
* Workstation GPU PC for:

  * teleoperation
  * dataset processing
  * Dreamer-4 training
  * live policy deployment

---

## **2.2 ROS 2-Based Control Stack**

The university uses a Docker-based ROS 2 Humble workspace containing:

### **Core Components**

| Component                 | Purpose                                                                        |
| ------------------------- | ------------------------------------------------------------------------------ |
| **spot_driver**           | ROS wrapper around Spot SDK (RobotState, ImageService, CommandService, leases) |
| **spot_nav_node**         | Publishes velocity/relative waypoint commands                                  |
| **NMEA UDP → ROS bridge** | Localization from Teltonika router                                             |
| **Exposed topics**        | `/spot/status`, `/spot/odom`, `/spot/camera/*`, `/spot/cmd_vel`                |
| **Exposed services**      | Spot navigation, posture, and safe operation services                          |

### **Why use ROS instead of the raw SDK?**

* Multi-sensor integration
* Easy teleop + logging
* Fast ML pipeline connections
* High modularity (your RL node is just one ROS node)
* SDK safety features handled by `spot_driver`

**Your Dreamer 4 policy outputs ROS commands.**
These are forwarded by the driver into the SDK and onto the real robot.

---

# **3. Project Workflow**

## **Step 1 — Teleoperation Dataset Collection**

### **Sensors and Data to Record**

For each timestep *t*:

| Category           | Data                                               |
| ------------------ | -------------------------------------------------- |
| **Vision**         | RGB cameras (front, ring), optional depth          |
| **Proprioception** | joint positions, velocities, foot contacts         |
| **Localization**   | odometry, IMU                                      |
| **Actions**        | teleoperation velocity commands (v_x, v_y, ω_z)    |
| **Metadata**       | waypoint target, episode boundaries, failure flags |

### **Logging via ROS 2**

```bash
ros2 bag record /spot/status /spot/odom /spot/camera/* /spot/cmd_vel
```

### **Frequency**

* Images: **5–10 Hz**
* State/action: **10–50 Hz**

### **Dataset Size**

* Minimum: **100–300 episodes**
* Strong: **500–2000 episodes** (10–50 hours of data)

---

## **Step 2 — Preprocessing Pipeline**

* Sync timestamps
* Resize / normalize images (e.g., 84×84 or 128×96)
* Build sequences (length 50–200)
* Normalize proprioception
* Store in Dreamer-ready format (HDF5 / TFRecords / npz)

Output: **offline training dataset**

---

## **Step 3 — Dreamer-4 Training**

Dreamer-4 stages:

### **1. World Model Learning**

Learns latent dynamics ( p(z_{t+1} | z_t, a_t) )

Trained on the **entire teleop dataset**.

### **2. Behavior Cloning Warm-Start**

Policy initialized from teleop actions.
Improves stability and reduces offline RL extrapolation error.

### **3. Model-Based RL (Imagination Rollouts)**

Actor/critic optimized from **latent imagined trajectories**.

### **Training Tips**

* Strong augmentations on images
* Conservative policy updates
* Shorter imagination horizons (15–50 steps)
* Replay real data continuously

---

## **Step 4 — Deployment Through ROS 2**

The trained policy is wrapped into a ROS node:

`dreamer_policy_node.py` or `dreamer_policy_node.cpp`

### **Subscriptions**

* `/spot/status`
* `/spot/odom`
* `/spot/camera/frontleft/image`
* (or whichever camera you choose)

### **Inference Output**

Dreamer produces a vector:

```
[vx, vy, yaw_rate]
```

### **Command Interface (Your Choice)**

#### **Option A — Publish directly**

```bash
/spot/cmd_vel   (geometry_msgs/Twist)
```

#### **Option B — Use navigation services**

Provided by `spot_nav_node`, e.g.:

```
ros2 service call /spot_nav/relative_move \
  spot_msgs/srv/Move "{dx:0.3, dy:0.0, dyaw:0.1}"
```

Either is valid — the thesis will use **high-level velocity commands**, safest for RL.

### **Safety Layers**

* command clamping
* tilt/IMU watchdog
* soft stop if policy outputs become unstable
* emergency stop monitoring from `spot_driver`

---

## **Step 5 — On-Robot Testing (Safe)**

### **Procedure**

1. Start in slow-speed mode
2. Use fixed, simple layouts
3. Expand environment complexity gradually
4. Increase autonomy level (human supervision → partial → full)

### **Rules**

* hard velocity bounds
* fallback “sit” command on anomaly
* geofence boundaries
* always keep physical E-stop active

---

## **Step 6 — Iterative Improvement**

* Add more demos where policy fails
* Retrain Dreamer with the expanded dataset
* Improve generalization
* Iterate until robust performance emerges

---

# **4. Observation & Action Specifications**

## **Observations**

* Resized RGB image(s)
* Proprioception vector
* Relative goal direction
* Optional: depth or LiDAR

## **Actions**

* High-level twist control:

  * ( v_x )
  * ( v_y ) (optional sideways)
  * ( \omega_z )

## **Control Rate**

* **5–20 Hz** (Matches ROS + Spot safety layer)

---

# **5. Reward Structure**

## **Sparse Reward**

+1 for reaching waypoint.

## **Dense Rewards**

* positive progress toward goal
* small step penalty
* smoothness penalty (optional)

## **Safety Penalties**

* collision
* excessive tilt
* emergency stop
* entering restricted zones

---

# **6. Safety & Infrastructure**

* E-stop registration
* hard velocity/acceleration limits
* geofence boundaries inside cage
* fallback posture command
* continuous episode logging
* optional simulation tests

---

# **7. SDK + ROS Practicalities**

While the ROS nodes handle the SDK internally, the thesis uses:

### **ROS to log:**

* images
* robot state
* teleop commands
* odom
* sensor readings

### **ROS to command:**

* `/spot/cmd_vel`
* navigation services

The underlying driver converts these into Spot SDK calls (RobotCommandService, ImageService, RobotStateService).

---

# **8. Evaluation Protocol**

## **Metrics**

* success rate
* time to goal
* collision count
* energy/path efficiency
* generalization to new obstacle layouts
* sample efficiency

## **Held-Out Test Scenarios**

* unseen obstacle placements
* lighting variations
* shifted goal positions
* minor disturbances

---

# **9. Experiments & Ablations**

Examples:

* BC-only vs Dreamer-4
* proprio-only vs vision+proprio
* depth vs no depth
* dataset size variations
* imagination horizon variations
* action space choices

---

# **10. Known Pitfalls & Solutions**

| Pitfall                   | Mitigation                           |
| ------------------------- | ------------------------------------ |
| Overfitting to visuals    | heavy augmentations                  |
| Dataset too small         | more demos, add random layouts       |
| Unsafe finetuning         | strict velocity clamps               |
| Offline RL divergence     | BC warm-start + conservative updates |
| World model hallucination | more real data, shorter rollouts     |

---

# **11. Dreamer-4 Implementation Details**

### **Hyperparameters**

* latent size: 256–512
* imagination horizon: 15–50
* batch size: 32–64
* LR: 1e-4 to 3e-4
* seq length: 50–200

### **Training Strategy**

1. strong world model pretraining
2. BC warm-start
3. incremental RL losses
4. optional online finetuning

---

# **12. Suggested Thesis Outline**

1. **Introduction**
2. **Background** (Dreamer, MBRL, Spot, ROS)
3. **System Setup**
4. **Dataset Collection**
5. **Model Architecture**
6. **Training Pipeline**
7. **ROS Deployment**
8. **Experiments & Results**
9. **Discussion & Limitations**
10. **Conclusion**
11. **Appendix** (configs, ROS graphs, code)

---

# **13. Startup Checklist**

* [ ] Build teleop logger
* [ ] Collect pilot dataset
* [ ] Implement preprocessing
* [ ] Train preliminary Dreamer-4
* [ ] Build ROS inference node
* [ ] Safe cage testing (low speed)
* [ ] Full dataset collection
* [ ] Final model training
* [ ] Thesis writing + evaluation

