#! /usr/bin/env python

# All the necessary libraries
import rospy
import time
# from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from move_robot import MoveRobot

# All Necessary Flags
IsBotNearObstacle = False


# CallBack Method for Laser Values Subscriber

def scan_callback(msg):
    global front_laser_range
    global right_laser_range
    global left_laser_range
    global front_min_value, right_min_value, left_min_value
    global idx_left, idx_front, idx_right

    ranges = msg.ranges

    # Front laser values start from  -5 deg  to  +5 deg  (Our Convention)
    front_laser_range[:5] = msg.ranges[5:0:-1]
    front_laser_range[5:] = msg.ranges[-1:5:-1]

    # Right Laser Values start from 300 to 345 deg  (Our Convention)
    right_laser_range = msg.ranges[300:345]

    # Left Laser Values start from 15 deg to 60 deg (Our Convention)
    left_laser_range = msg.ranges[60:15:-1]
    '''
    min_range, idx_range = min((ranges[idx_range], idx_range) for idx_range in range(len(ranges)))
    '''
    front_min_value, idx_front = min(
        (front_laser_range[idx_front], idx_front) for idx_front in range(len(front_laser_range)))
    right_min_value, idx_right = min(
        (right_laser_range[idx_right], idx_right) for idx_right in range(len(right_laser_range)))
    left_min_value, idx_left = min((left_laser_range[idx_left], idx_left) for idx_left in range(len(left_laser_range)))


# Initializing all the laser scan variables

front_laser_range = []  # List of Values for Laser front
right_laser_range = []  # List of Values for Laser right
left_laser_range = []  # List of Values for Laser left

front_min_value = 0  # Minimum Laser Value for Front
idx_front = 0  # Front Index for Iterating range of values
right_min_value = 0  # Minimum Laser Value for Right
idx_right = 0  # Right Index for Iterating range of values
left_min_value = 0  # Minimum Laser Value for Left
idx_left = 0  # Left Index for Iterating range of values

# Creating the Main Node
laser_scan_subscriber = rospy.Subscriber('/scan', LaserScan, scan_callback)
rospy.init_node('maze_solver')
bot_movement = MoveRobot()
# Initializing the bot
bot_movement.send_cmd(0.0, 0.0)

rate = rospy.Rate(10)
time.sleep(1)  # Delay for Initializing node

print ("Starting the Movement..")
bot_movement.send_cmd(0.1, -0.5)
time.sleep(2)


while not rospy.is_shutdown():
    while not IsBotNearObstacle and not rospy.is_shutdown():
        if front_min_value > 0.2 and right_min_value > 0.2 and left_min_value > 0.2:
            bot_movement.send_cmd(0.15, -0.1)
        elif left_min_value < 0.2:
            IsBotNearObstacle = True
        else:
            bot_movement.send_cmd(0.0, -0.25)

    else:
        if front_min_value > 0.2:
            if left_min_value < 0.12:
                bot_movement.send_cmd(-0.1, -1.2)
            elif left_min_value > 0.15:
                bot_movement.send_cmd(0.15, 1.2)
            else:
                bot_movement.send_cmd(0.15, -1.2)
        else:
            bot_movement.send_cmd(0.0, -1.0)
            while front_min_value < 0.3 and not rospy.is_shutdown():
                pass
    rate.sleep()
