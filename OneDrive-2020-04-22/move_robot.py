#! /usr/bin/env python

import rospy
from geometry_msgs.msg import Twist


class MoveRobot:
    def __init__(self):
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.move_msg = Twist()

    def send_cmd(self, linear=0, angular=0):
        self.move_msg.linear.x = linear
        self.move_msg.angular.z = angular
        self.pub.publish(self.move_msg)
