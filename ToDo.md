- make a planning for phases of thesis              CHECK
- Prepare connecting to SPOT                        CHECK
- Look at how we are going to record data           CHECK
- prepare dreamer model and test training loop      CHECK
    -> JAX faster than pytorch
    ->
- study dreamer v3                                  CHECK


Data collecting to Do:
- look if we can find all the right data to record  CHECK
- record while teleopping for first data.           CHECK
- see how we can get obstacle map                   CHECK
- try to go over obstacles with autowalk 


28/1:
- Try to connect to localgrid obstacle map via python SDK                                           CHECK
- Check all data we need again                                                                      CHECK
- dive into info of topics, nodes, services                                                         CHECK
- decide if custom gait is worth it                                                                 CHECK
    - switch to action space: velocity and rotation (focus on stability? and decision making)       CHECK


29/1:
- Joris mailen dat gaits aanpassesn lastig is en nieuwe action space voorstellen                    CHECK
- define the actual action/observation/information space.                                           CHECK
- 


30/1:
- try terrain local grid at inference speed -                                                       CHECK
    > look at streaming service: https://dev.bostondynamics.com/python/examples/visualizer/readme   
    ACHIEVED: 6.3 Hz with python script.
    - otherwise go with depth map


- first data runl
- use fiducials to walk around  
- check position odometry accuracy                                                                  CHECK
- find gait commands service                                                                        CHECK

- try spot control driver to see the actions we need to send to the SPOT when training.             CHECK
    ros2 launch spot_driver spot_driver.launch.py [config_file:=<path/to/config.yaml>] [spot_name:=<Spot Name>] [tf_prefix:=<TF Frame Prefix>] [launch_rviz:=<True|False>] [launch_image_publishers:=<True|False>] [publish_point_clouds:=<True|False>] [uncompress_images:=<True|False>] [publish_compressed_images:=<True|False>] [stitch_front_images:=<True|False>]

- solve aiortc docker build error and rebuild                                                       CHECK


3/2: 
- discuss final collection plan with Joris                                      CHECK
    - we will use the gait as well. 
    - perhaps only use terrain grid but definetely use it
    - reward function with vel vector and position is fine with odometry. maybe manuually give error after each episode
    - 
- check recording data completely
    - stitch images via node and subsribe to stitched node instead.             CHECK
    - Include gait yes or no                                                    CHECK
    - check odometry error                                                      
    - Check how to read out locomotion_mode                                     CHECK
    - Check if action and odometry matches.                                     CHECK
    
- David Abbink nog een keer mailen                                              CHECK



4/2:



9/2:
- email bjorn to see what data he might need                                    CHECK
- check response image stitcher issue solve                                     CHECK  
- finish collection scripts                                                     CHECK
- finish teleop with ps4 controller                                             CHECK
- david abbink nog maals mailen ter bevestiging?                                CHECK
- is obstacle map available at inference time?                                  CHECK
- rebuild docker image                                                          CHECK

10/2
- begin writing introduction and method, data collecting background and decision, observation space, action space.
- email to get explanation of cage use wednesday                                CHECK
- build recording loop                                                          CHECK



11/2 
- test teleop with ps4 controller                                               CHECK
- insert z height of robot body in data                                         CHECK (not possible via driver)
- test recording loop with ps4 controller                                       CHECK
- collect first full data rosbags                                               
- look at obstacles for setup
- add pillow to container.                                                         







