# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:23:17 2019

@author: tony
"""
import matplotlib.pyplot as plt
import rospy
import ltv_contouring_mpc as ltvcmpc
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from mpcc_msgs.msg import MPCC_Params

history_x = []
history_y = []
history_v = []

def paramCallback(msg):
    print(msg.weights.q)

def velCallback(msg):
	v_cmd = msg.linear.x
	history_v.append(v_cmd)
def odomCallback(msg):
    
    # Get the x, y and theta measurements (states) and update xi_curr accordingly
    x_measure = msg.pose.pose.position.x
    y_measure = msg.pose.pose.position.y
    # v_measure = msg.twist.twist.linear.x
        
    history_x.append(x_measure)
    history_y.append(y_measure)
    # history_v.append(v_measure)

def main():
    # Filename for track
    track_points_filename = 'data/track_optimal.mat'
    
    # Load track 
    track_points = ltvcmpc.load_track(track_points_filename)
    
    # Initialize the ROS node
    rospy.init_node('MPCC Data Logger')

    # subscribe to the /odom topic and run callback function
    rospy.Subscriber("/odom", Odometry, odomCallback) 
    
    # subscribe to the /odom topic and run callback function
    # rospy.Subscriber("/cmd_vel_mux/input/teleop", Twist, velCallback) 

    # subscribe to the /odom topic and run callback function
    rospy.Subscriber("/rover/mpcc_params", MPCC_Params, paramCallback) 

    rospy.spin() # runs forever
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(track_points['outer'][0,:], track_points['outer'][1,:], 'k')
    ax.plot(track_points['inner'][0,:], track_points['inner'][1,:], 'k')
    im = ax.scatter(history_x, history_y, c=history_v, cmap = 'RdYlGn')
    fig.colorbar(im, ax=ax)
    fig.savefig('heatmap.png')
    
if __name__ == '__main__':
    main()
        
