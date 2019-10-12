#!/usr/bin/env python

from ltvcmpc_msgs.srv import InitPosition,InitPositionResponse
import rospy

def handle_init_position(req):
    print "x = %s, y = %s"%(req.a, req.b)
    return AddTwoIntsResponse(req.a + req.b)

def add_two_ints_server():
    rospy.init_node('add_two_ints_server')
    s = rospy.Service('add_two_ints', AddTwoInts, handle_add_two_ints)
    print "Ready to add two ints."
    rospy.spin()

if __name__ == "__main__":
    add_two_ints_server()