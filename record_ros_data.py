#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from can_translator.msg import Steering
import pickle

image = None
steering = None

def image_callback(data):
    image = data.data

def angle_callback(data):
    steering = data.steering_angle

def getID(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/camera/left/image_raw", Image, image_callback)
    rospy.Subscriber("/vehicle/steering", Steering, angle_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spinOnce()
    rospy.sleep(0.2)

    # write image and steering angle to pickle file
    if image is not None and steering is not None:
        xy = (image, steering)
        ID = getID()
        with open('rosbag_data/' + ID, 'wb') as f:
            pickle.dump(xy, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    listener()