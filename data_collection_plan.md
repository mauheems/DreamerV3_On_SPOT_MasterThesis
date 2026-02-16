# SPOT + DreamerV3 Navigation Plan

## 1️⃣ Spaces

**Action space:**

* Velocity XY + yaw — standard for quadruped locomotion.
* Optionally: stance height (not needed initially, SPOT switches gaits automatically).

**Observation space (camera):**

* Front camera (pixels) 
* **Goal Vector:** Relative distance and direction (sin/cos angle) to the target coordinates. (Computed in wrapper).

**Information space (privileged):**

* Obstacle height map (local grid) — kept as privileged info.
* World model uses it to learn physics and environment layout.
* Velocities / odometry — optional, can be included as privileged info for world model.

✅ **Conclusion:** Obstacle grid remains **information space**. Goal vector is an **observation** (actor needs it to navigate). Observation = what robot realistically perceives to perform the task.

---

## 2️⃣ Data Collection

* **Strategy:** Record raw data (images, odometry, terrain) without hardcoded rewards.
* **Trajectory generation:** Teleop / Autowalk to create diverse paths around obstacles.
* **Metadata:** Log global Start and End positions (from Odometry) for every episode.

**Per-timestep logged data:**

* Front camera pixels (Stereo/RGBD)
* Actions (XY velocity + yaw)
* Raw Odometry (Global X, Y, Yaw)
* Terrain Grids (Height maps)
* Termination reason (if manual override/crash)
* Optional: gait


✅ **Design Decision:** Rewards and Goal Vectors are **not** recorded. They are calculated **post-hoc** in the training wrapper using the logged Odometry. This allows tuning the task (e.g., "Go 5m" vs "Go 10m") without re-recording data.

---

## 3️⃣ Reward Design (Computed Post-Hoc)

**Dense per-timestep rewards:**

1. **Progress toward goal:** `(dist_to_goal_t-1 - dist_to_goal_t)`. Provides velocity toward goal, rewarding detours if they eventually lead to the target.
2. **Success/failure:** Large sparse reward when within X meters of goal, penalty if stuck/crashed.
3. **Collision / Smoothness:**
    * Negative reward if velocity drops near 0 while commanding movement.
    * Negative reward if `Info_Terrain` shows obstacle < 0.5m away.
    * Odometry smoothness penalty.


---

## 4️⃣ Termination Signals

* **Goal reached:** Fiducial nearly fills front camera.
* **Collision / fall:** Emergency termination.
* **Timeout / stuck:** Optional if robot hasn’t moved sufficiently over N seconds.

These signals are required for:

* World model latent resets
* Value bootstrapping for critic
* Defining success/failure outcomes

---

## 5️⃣ Dataset Checklist

| Component          | Logged / Defined?                                        |
| ------------------ | -------------------------------------------------------- |
| Actions            | XY velocities + yaw ✔️                                   |
| Observations       | Front camera ✔️                                          |
| Privileged info    | Obstacle height map, odometry ✔️                         |
| Reward components  | Progress, speed, success/failure, collision ✔️           |
| Termination flags  | Goal, collision, timeout ✔️                              |
| Episode metadata   | Start / end / episode ID ✔️                              |
| begin/endpoint     | Used for labeling progress ✔️                            |
| Diversity          | Autowalk/teleop to generate partial & failed attempts ✔️ |

---

This forms a **modular and flexible dataset design** compatible with DreamerV3 / Informed POMDP, while keeping actor observations realistic and world model learning rich latent dynamics.


