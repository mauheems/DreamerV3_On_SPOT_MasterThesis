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
- collect first full data rosbags                                               CHECK
- add pillow to container.                                                      CHECK 


17/2 
- look at obstacles for setup                                                   CHECK  
- make first run                                                                CHECK
- check reward function                                                         CHECK
- collision flag on recorder?                                                        
- make 'delete' episode and continue button on recorder                         CHECK
- check frequency of rosbags.                                                   CHECK  


18/2:
- include recording on joystick teleop                                          CHECK
- merge launch files recording and joystick                                     CHECK
- email bjorn GPU cluster                                                       CHECK
- more data collecting                                                          CHECK


19/2
- see if we can replay the image to see what happens of an rosbag               CHECK
- post process data to see if we have everything we need                        CHECK
    - normalize data (maybe reset everything before recording an episode.)                    


20/2:
- learn the AI cluster intro                                                    CHECK   


23/2
- check teleop + recorder.                                                      CHECK
- see if we have to reinitiate the terrain height map client every recording    CHECK
- check recording rate. Now it seems 3 Hz only?                                 CHECK
    and fix it
- collision detector by button??                                                CHECK

24/2
- preprocess data fully: gait one = gait two in data.                           CHECK

25/2:
- check spot env wrapper and rewards via gemini/chatgpt                         CHECK
- call python train.py for cluster                                              CHECK
- first training run complete                                                   CHECK


26/2 
- check hz of topics: ros2 topic hz /camera/frontmiddle_virtual/image           CHECK
- check node builds of terrain map and image stitcher                           CHECK
- make very bad recording to amplify penalties and reward for testing           CHECK
- get rewards to line up with events                                            CHECK

27/2
- odometry speed is calculated in world frame!!! this is why velocity doesnt match commands.            CHECK


2/3:

- how does the velocity command of trot 2m/s line up with velocity comman of crawl 1m/s                 CHECK
- do the actions line up of dataset and spot.py, also gait selection -> def _action is not defined      CHECK
    -> look at minecraft_base.py


3/3:
vragen:
- retrieve password for DAIC                                                                            CHECK
- write methodology                                                                                     CHECK

6/3:
- First DAIC training                                                           CHECK

9/3:
- make a recording with all events                                             CHECK
    - gait switch
    - quick straight part
    - walk into an object for long period
    - force imbalance on foam
    - note down events
- make as many data as possible                                                CHECK
- start training with no priviliged info                                       CHECK 
- redownload docker image spot driver/Pillow/scipy                             CHECK

10/3:
- put image through encoder                                                    CHECK
- compare training runs informed/           uninformed                                    CHECK
- see if terrain can be decoded from latent state                                       CHECK
- look at feedback paper                                                                CHECK

16/3:
- test deployment logic                                         CHECK
- deploy a policy on the spot                                   CHECK
- write some paper
- add improvements to training cycle
- set gait command to trot and crawl                            CHECK
- check why uninformed works as expected but informed not       CHECK
- check if raw odometry is world frame                          CHECK

17/3:
- decode images of uninformed and informed to see difference. we need to see world model. CHECK
- check if info_terrain can be observation space without 'info_' tag        CHECK
- check if informed terrain map goes through decoder fine.                      CHECK
-


31/3:
- evaluate deconstructed terrain map as well                                CHECK
- check with master coordiantor for final checklist and colloquia           CHECK
- joris sijs mailen voor morgen.                                            CHECK
- fix terrain input for world model notebook.                               CHECK



1/4:
- evaluate the actor critic in the dream. chatgpt chat!!                CHECK
- check gait input reconstructed as well                                    

2/4:
- train smaller rssm models for deployment as well                                          CHECK
- get last trainings from DAIC                                                              CHECK
- check how we would the policy directly on the robot instead of laptop                     CHECK
- check if black observations during deployment causes model to go backwards. 
    - Check different episodes in actorcritic notebook
- build in reset for policy node wihtout reloading policy but resetting target goal         CHECK
- see if we can fix sensor fialure issue between policy checks                              CHECK
- IMPORTANT: vx has 2 m/s limit but vy ofcourse not!! this needs to be trained again.       CHECK
- think about removing crawl for now. too much noise for the model.                         CHECK
- update docker image                                                                       CHECK
- put checkpoints on drive

5/4:
- check actor critic quality with smaller horizon
- check larger batch length  performance with working config 



8/4:
- check if xlarge rssm can be deployed now                          CHECK
    - check if params float 16 for jax helps this problem 
- check if vy command does not overreach 1 m/s                      CHECK
- check if smaller rssm models give decent performance              CHECK
- check if new physics loss imroves performance.                    
- penalty for high velocity after end goal is reached.              
- computation to position because this is also the rewardfuncton

9/4: 
- check if checkpoints are complete.                        CHECK
- check checkpoint compressor and convert all               CHECK
- make batch size 64 checkpoints.                           CHECK
- remake rewards for getting close to the end goal          CHECK
- find out why training is slow                             CHECK
- remove multi step consistency if that helps               CHECK
- put the raw checkpoints on hrd drive                      CHECK


13/4: 
- test bootstrap decrease
- test other episodes for current checkpoints.
- test removing position and redundant observations
- see how to lower critic loss. and what this exactly means in the math
- see how to lower the critic value prediction of 300 vs 0.5 imagined reward (what do these values mean)                                                    CHECK
- see difference in rewards                              CHECK










Suggestions for improvement of offline dreamer:
- penalize the values of states that the World Model hasn't seen in the dataset. (vel too high)
- data augmentation
- penalize action if its too far away from what the data did (conversatism)
- since we are giving the goal position relative to the robot frame our position and orientation in world frame are redundant as observations and possible causing error for the networl
- implement loss for decoded prior | decoded posterior instead of in latent space

Suggestions for results:
- to proof image/terrain as observations enable meaningful information. run the training without images and show performance drop.!!
- to prove it has better policy performance we need to test in actual environment succes rate.
    - policy with and without images/terrain
    - policy and mask the images to see performance drop
    - policy testen op andere ondergrond als foam
- ablation to prove that physics consistency loss increases prior performance.
- 