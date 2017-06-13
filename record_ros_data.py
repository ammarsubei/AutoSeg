#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CompressedImage
from can_translator.msg import Steering
import pickle, string, random
import cv2

image = None
previous_image = None
steering = None

def image_callback(data):
    global image
    image = data.data
    print(data.data)
    #print("Got image!")

def angle_callback(data):
    global steering
    steering = data.steering_angle
    #print("Got steering!")

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

    # write image and steering angle to pickle file
    global previous_image
    while True:
        # spin() simply keeps python from exiting until this node is stopped
        rospy.sleep(0.2)
        if image is not None and steering is not None:
            cv2.imshow('Image', image)
            cv2.waitKey(1)
            xy = (image, steering)
            ID = getID()
            with open('rosbag_data/' + ID, 'wb') as f:
                pickle.dump(xy, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(ID)
            if image == previous_image:
                break
            else:
                previous_image = image

if __name__ == '__main__':
    listener()
