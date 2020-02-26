# Optimal Path Planning and Following for Autonomous Race Cars - ROS Python Package
ROS implementation of an mpc controller for Aion R6 Rover- Final Year Project: Optimal Path Planning and Following for Race Cars

## Video Results
All video results of the project can be found [HERE](https://github.com/khewkhem/autonomous-race-cars)

## Contributors
 * Seth Siriya
 * Tony Srour
 * Khoa Tran

## Acknowledgements
 * Supervised by Dr. Ye Pu, A/Prof Iman Shames and Prof Michael Cantoni.
 * This implementation draws from [Optimization-Based Autonomous Racing of 1:43 Scale RC Cars](https://arxiv.org/abs/1711.07300).
 
 # Simulation Instructions: 
 
 ## Requirements
  * Ubuntu 16.04 OS
  * ROS Kinetic installed
  * Mavros installed
  * rospkg, pyyaml, matplotlib, osqp installed on python 3.5+
  
 ## Steps
  1. Create a new catkin workspace in your home directory named "YP2_capstone_ws" (follow the ROS [tutorials](http://wiki.ros.org/ROS/Tutorials) if unfamiliar with ROS)
  2. Install the provided ROS packages using catkin_make
  3. Run the launch file in the ltvcmpc_gazebo_env package
  4. Run the mpcc_controller_sim launch file in the ltvcmpc_controller package (you can edit the mpcc_control_node executable to change the weights or constraints of simulation)
  5. the rover will now complete one lap if all instructions are followed correctly, plots and results can be found in data_plotter/plots/simulation
