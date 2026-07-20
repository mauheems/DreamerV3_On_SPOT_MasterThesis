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
- penalty for high velocity after end goal is reached.              CHECK
- computation to position because this is also the rewardfuncton    CHECK

9/4: 
- check if checkpoints are complete.                        CHECK
- check checkpoint compressor and convert all               CHECK
- make batch size 64 checkpoints.                           CHECK
- remake rewards for getting close to the end goal          CHECK
- find out why training is slow                             CHECK
- remove multi step consistency if that helps               CHECK
- put the raw checkpoints on hrd drive                      CHECK


13/4: 
- test other episodes for current checkpoints.              CHECK
- test removing position and redundant observations         
- see how to lower critic loss. and what this exactly means in the math CHECK
- see how to lower the critic value prediction of 300 vs 0.5 imagined reward (what do these values mean)                                                    CHECK
- see difference in rewards                              CHECK

14/4:
- test bootstrap decrease
- check if new physics loss imroves performance.           
- check if new rewards lets robot stay near end         NOPE
- check if negative x target goal turns robot around.   NOPE      
- IMPORTANT what if fully offline does not work?        CHECK
- and how would we finetune online??
- is the critic bad or are the dreams bad?              BOTH


15/4:
- test lower imagination horizons for training         CHECK 
- finish NoObs dataset                                  CHECK
- train on NoObs dataset                                CHECK
- do we need images even for obstacles.                 CHECK


16/4:


17/4:
- test noobs dataset.                                   CHECK
- test low imag checkpoints                             CHECK
- recompute rewards for noobs and check them            CHECK                        
- train online with noobs       
- Fix how to finetune model on SPOT online              
- train on terrain only
- train posterior critic
- look at warm up during training     
- remove redundant velocities for noobs training        CHECK
- see if we can change the task to action space.


20/4:
- test the most basic obs space trained model and look at latent drift
- discuss possible thesis topics
    - Quantifying Latent Drift in RSSM-Based World Models
    - Decoupling World Model Quality from Policy Performance in DreamerV3
    - From Vision to State: Does Observation Modality Affect World Model Stability?”.

21/4: 
- reconvert the dataset noobs.          CHECK
- tune rewards                          CHEck
- recompute rewards                     CHECK
- retrain                               CHECK


- static target goal'
- add data for walking backwards and not reaching the target CHECK
- add penalty when reach the goal and move further.          CHECK
- record rosbag during policy deployment if anything weird happens

22/4:
- ablation for static goal position                                             CHECK
- ablation for horizon on goal position                                         CHECK
- ablation on position including in obsrvation and see what goal target does    CHECK
- ablation xlarge rssm small actorcritic                                        CHECK
- check mounting of harddrive for online finetuning
- check flow of online finetuning

23/4:
-- find out why reconstrcution is worse during deployment then during offline test
    - does odometry not track the target goal well enough
- find out what is the advantage of online learning and filloing the repaly buffer  CHECK
- retrieve the rewards from the rosbag now instead of recordings                    CHECK
- Why do the models think that going backwards is getting it reward -> even if target reduces wrongly, why not forward then                                                         CHECK


28/4:
- look at reward function again             CHECK
- look at freshly trained things again      CHECK
- fix vel input in world model input.       CHECK


29/4:
- check if orientation is 0,0 at start of deployment            CHECK
- next testing should be without smoothing                      CHECK
- remember what the replay buffer actaully contains              CHECK
    - increase randomization of target goal place on trajectory to see if this increases reward accuracy and stoppage when goal is reached.       
- implement smoothng not by limiting jerk but by averaging the commands to reduce noise     CHECK
- does the robot know how to stand still? is this included in data?                     CHECK
- might need to actaully put high penalty using any command when reaching the goal      CHECK
- fix deployment notebook for the many graphs it has.                                       

- 10:58:    check observations after goal has been reach
            check rewards after goal has been reached        
- 11:03 : smoothing off
-11:33 : new smoothing   


30/4: check timing issues of deployment 

6/5: 
- check if continuation flag is given to deployment or not.   CHECK

7/5:
- look at training data from last runs                          CHECK
- decide if we need continuation signal                         CHECK
- fetch NEW DATA for standing still at the targetposition       CHECK
- retrain WITH termiantion signal                               CHECK
- fix container gpu access!!                                    CHECK
- determine succes graphs for paper and set up notebook         CHECK

8/5:
- look if training inference is sereously lacking sending a correct target position and if higher hz helps. 
    - now i think the rssm call i very slow live.
- check new deployment observations
- set up agent critic notebook for self set first observation


12/5
- set greenlight meeting and ask about defence data   CHECK
- retrain with new rewards v6                           CHECK
    - fix orientation
    - no big out of radius penalty, keep distacne to goal penalty before and afte goal reached
    - remove time penalty
    - remove jerk penalty at goal radius
- mail joris over vakantie 10 juli                      CHECK

19/5:
- 
-

19/5:
- train with more data                                      CHECK
- discuss contributions of paper
- discuss results of paper and ask joris to check paper
- show working policy

21/5:
paper:
    - start training for actual result graphs and setup the sbatch arrays   CHECK
    - make figures for replay buffer and imagination rollout explanation prior, posterior CHECK
    - imagination rollout drift figure should continue further than trained on to show error compounding        CHECK

22/5: 
paper: 
    - generate results graphs of imagination horizon ablation and rssm size ablation
    - show regions in the latent space 2D graph
    - one read-through to take out inconsistent terms and redundant wording    

- record a long episode of 100 horizon to test horizon degredation. CHECK
- record deployment bags of unsuccesful policies.                   CHECK
- make picture of setup                                             CHECK

25/5:
- train again with orientation penalty                              CHECK
- rewrite imagination results and experiments                       CHRCK
- really look for ai written pieces. go through once                
- add the current visuals 

26/5:
- finish the architecture diagram and matching text
- SPOT dynamics figure
- DO include RSSM size and imagination horizon but only to show that rssm size lowers loss but not mostly increases noise. and imagination horizon only influences the data distribution to end goal behaviour. 
- find the missing figures 

27/5:
- find failure mode graphs and finish results text  CHECK
    - bad posterior
    - bad imag horizon
    - cluttered projection with inserted deployment episode CHECK
        - processed_data_NoObs_with_rewards_v4 with rewardsv4_A_dyn1_rep005_stoch32_05-07
    - reward hacking: being at the end goal according to reconstruced observation despite velocity observations pushing it further out. getting reward because it thinks its staying at the end goal
        - realtd: 1,2 CHECK
        - rewardsv2 medium dyn rep 
    - reward exploitation: going out of the target radius to get a new reward because our reward function rewards re entry when overshooting. we edited the reward function to transition back after overshoot
        - rewardsv9_2026-05-25_10-37_S12 
    - failure mode during actor critic training of staying at goal for to long shifting data distribution to onyl staying at end goal.
        - rewardsv9_2026-05-25_10-35_S16_base CHECK

- make trajectory graph with orientation angles to show orientation reward works as well.
    - rewardsv9 S10 rosbag 0 succesful trajecotry       CHECK

28/5:
- finish discussion CHECK
- make results trajectory graphs clean  CHECK
- finish SPOT and projectiarchitecture figures and insert them into text CHECK
- finish conclusion CHECK

-  edit reward function CHECK

29/5: 
- check all figures again if there are better one's. dont mind the style too much for now.  CHECK
- check whole paper for AI
- shape the paper in overleaf, figures and formulas

## To DO PAPER: ##
- reconstruction and prediction losses need to be explained better ✅
- maintain consistent terminology with reconstructed, decoded, predicted and imagined ✅
- move augmented target position data to preprocessing, duh ✅
- explain the future modes reward exploitation, reward hacking etcc ✅
- where to explain KL divergence ✅
- fix graphs that say the wrong terms like decoded, predicted, reconsructed etc. ✅
- fix offset figure 5       ✅
- check new section in results if that makes sense ✅
- do our definitions of latent consistency and ...  actually contribute somewhere. ✅
    - define imagination consistency and prior consistency ✅
- what do we call a prior rollout without policy.  search " Because the actions stay the same in both rollouts, differences between the posterior reconstruction and prior rollout show inaccuracies in the learned latent dynamics and our"✅ 
- do a full run while keeping in mind terms
    - prior, posterior, imagined, decoded, reconstructed
    - shorten world model architecture and check logic again 
    - check all the comments again
- check if 'latent state' needs to be changed to stochastic state. 
    - latent state is actually det and stoch together.
    - stoch state can be stoch latent state
    - deter state is hidden state
- think of title 
- zoom in on figure 11
- increase letter size architecture diagram and make text like 'enc' 'dec' 'h1' more contrast
- figure 3 highlight the argument in the graph please
- figure 10 should include extra region for counter lingering
- align introduction with changes in paper
- send diploma information
- does every figure capton tell what rollout it is?
- discussion on pre deployment framework should include actual predeployment analysis
- do not define it as a pre-deployment framework. its just a 'pre-deployment analysis'
- can we strengthen the dataset distribution claim somewhere because we do have this in the appendix
- ask acces GPU cluster beyond 1st june
- fix appendix graphs
- check the appenndices references in text
- appendix should include training curves instead of metrics
- more details in future work.
    - train world model and actor critic seperately

- full read
    - actor and critic function fix
    - check orientation quaternion
    - do we mention that our what succesful policy is linked to all the succesful graphs
    - are the failure modes 
    - move actor critic training failure mode to last probably
        - we could be more specific to link back to the imagination consistency and latent conistency
- check related work section
- fix all the indents, figure location and formulas in the compiled paper 
- check sources
✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ 


- discussion 
- check turnitin AI

- push code to github openbots autonavigation









deployment rosbags good and bad:
bad:
    - rewardsv8 H25 overshoots a lot and comes back
good:
    - S10,S16 are quick and dont overshoot much.













# graphs
Here is a complete style ruleset based on everything we've established:

---

**Figure style rules for paper-ready matplotlib plots:**

1. **No subplot titles** — use bold panel labels **(a)**, **(b)**, **(c)** as text annotations in the top-left corner of each subplot (`fontweight='bold'`, `fontsize=12`, `transform=ax.transAxes`, position `(0.02, 0.97)`, `va='top'`).

2. **Axis label format: "Description (unit)"** — always "what it is" first, then unit in parentheses. Examples: `Time (s)`, `Goal position (m)`, `Velocity (m/s, rad/s)`, `Reward (per step)`, `Distance to goal (m)`. Never a bare unit like `m` alone.

3. **Every subplot gets its own x-axis label** — do not use `sharex=True` and a single shared label. Each panel must be independently croppable.

4. **Legend always `loc='upper right'`** — no exceptions unless the data physically sits in the top-right, in which case use `'upper left'`.

5. **GT vs decoded line convention:**
   - Ground truth / actual: solid line (`lw=2.0`), labelled `GT <name>` (e.g. `GT vx`, `GT reward`)
   - Model reconstruction / decoded: dotted line (`ls=':'`, `lw=1.8`), same color as its GT counterpart, labelled `decoded <name>`
   - GT and decoded of the same channel always share the same color

6. **Color consistency** — use `plt.cm.tab10.colors` cycling by channel index, so the same channel is always the same color across GT and decoded. For single-channel plots (e.g. reward), use `color='C0'` explicitly for both lines.

7. **Figure size** — use `figsize=(14, 10)` for 3-panel vertical figures, scaled proportionally for other layouts.

8. **Grid** — always `ax.grid(True, alpha=0.3)`.

9. **Spine style** — `axes.spines.top: False`, `axes.spines.right: False` (set globally via `plt.rcParams`).

10. **Font size** — base `font.size: 11` via `rcParams`, legend `fontsize=8`, panel label `fontsize=12`.

11. **No `fig.suptitle`** — the figure title belongs in the paper caption, not in the figure itself.

12. **`plt.tight_layout()`** always called before `plt.show()`.

---

**Copy-paste prompt template:**

> *Apply these figure style rules: no subplot titles — use bold (a)/(b)/(c) annotations at top-left instead. Y-axis labels follow "Description (unit)" format matching "Time (s)". Every subplot has its own x-axis label so panels can be cropped independently. Legends always upper right. Ground truth lines are solid lw=2, labelled "GT \<name\>"; decoded/reconstructed lines are dotted lw=1.8 in the same color, labelled "decoded \<name\>". Grid alpha=0.3. No suptitle. tight\_layout before show.*