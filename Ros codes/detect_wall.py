#! /usr/bin/env python

import rospy
from move_robot import MoveRobot
from sensor_msgs.msg import LaserScan

class StopWall():
    def __init__(self):

        self.sub = rospy.Subscriber('/kobuki/laser/scan', LaserScan, self.callback)
        self.moverobot_object = MoveRobot()

    def callback(self, msg):
        print msg.ranges[360]
        
        if msg.ranges[319] > 1:
            linear_x = 0.5
            angular_z = 0.0
            self.moverobot_object.send_cmd(linear_x, angular_z)

        #If the distance to an obstacle in front of the robot is smaller than 1 meter, the robot will stop
        if msg.ranges[319] <= 1:
            linear_x = 0.0
            angular_z = 0.0
            self.moverobot_object.send_cmd(linear_x, angular_z)
            


if __name__ == '__main__':
    rospy.init_node('debug_day1_node')
    stopwall_object = StopWall()
    rospy.spin()