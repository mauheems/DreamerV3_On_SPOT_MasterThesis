# Master-Thesis
Adaptive Gait control by means of Dreamer Algorithm

This master thesis will try to implement dreamerv3 algorithm to let a SPOT quadruped robot make independent decisions on how to traverse terrain. 

To simulate a simple version of real terrain we will be working inside of a closed environmenet with boxes and small terrain obstacles. We will have to collect verious data before training to counteract imagination drift and OOD.

To do:
- collect 5-10 hours of data from random exploration to expert driven data
- train a informed dreamerv3 world model from this action, information and observation space
    - the observation space will be the frontcamera (pixels) and an obstacle map
    - the information space will be pixels, obstacle map velocity and gait
    - the action space will be velocity, yaw, and a gait parameter (1-4)

- train an actor critic on this world model, hopefully offline will work since we will use a informed model to counteract imagination drift and OOD
    - the reward function will include velocity, failed configurations, and if we make it from A to B

- get results and graphs
    - regular autowalk speed vs dreamerv3
    - reward function graphs over time
    - informed vs uninformed
    - graph of correct trajectories