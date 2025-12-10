# kick-off notes

- Is the hardware good enough for what we are trying to accomplish?
- Are we going for adaptive gait control such as in the paper. This can control phase, step size gait type etc. low-level control. or are we going for high level control that optimizes for traversing a forest. Do we need MPPI/MPC hybrid then?
- Maybe we should jsut focus on a forest dataset real-life world model creation and put dreamer on that to test? If the main goal is landscape/environment traversion. Maybe use dreamer purely for high-level planning and decision making. going from A to B and walking around trees, bushes, dead branches, purely on camera images/proprioception. SPOT can run on paths wiht no obstacles and knows to calm down with bushy areas. 
- no car available for driving with SPOT. Is a forest the most feasable then. Cant go to campus ground?
- MPPI is very computationally expensive and requires realtime loop. 
- Daydreamer (paper), does add online loop. bsed on dreamer v1 so might be nice to update that. or combine it somehow.


- kijk in de sdk wat er mogelijk is
- how much data do we need?
- Where?
- automatiseren.

https://dev.bostondynamics.com/docs/concepts/robot_services.html
Robot Services — Spot 5.0.1.2 documentation
 
https://github.com/OpenBots/openbots_backpack
 

FOUT 
r.66 RUN cd /home/ob/openbots_ws/src/externals/spot_ros2 && \

    ./install_spot_ros2.sh --arm64
 