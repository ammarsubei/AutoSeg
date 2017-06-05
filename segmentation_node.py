#!/home/revo/anaconda2/envs/ROS+TF/bin/python

from keras.models import Sequential, load_model
from keras.utils import plot_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D
import sys, time
import roslib, rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tensorflow as tf
import cv2

num_classes = 34
img_height = 720
img_width = 1280
img_size = (img_width, img_height)
input_shape = (img_height, img_width, 3)
batch_size = 4
cap = cv2.VideoCapture(sys.argv[1])
out = cv2.VideoWriter('output.avi',fourcc, 20.0, img_size)

model = autoseg_models.getModel(input_shape=input_shape,
                                num_classes=num_classes,
                                residual_encoder_connections=True,
                                dropout_rate=0.0)

model.load_weights('main.h5')

sgd = keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=pixelwise_crossentropy, optimizer=sgd, metrics=[pixelwise_accuracy])

class lane_finder:

    def __init__(self):
        self.model = model
        self.graph = tf.get_default_graph()
        self.image_pub = rospy.Publisher(segmentation,Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/left/image_raw",Image,self.callback, queue_size=1, buff_size=100000000)

    def callback(self,data):
        start = time.clock()
        try:
            raw_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        model_input = np.array([raw_img]).astype(np.float32) / 255
        with self.graph.as_default():
            model_output = self.model.predict(model_input)[0] * 255
            lane_lines = model_output.astype(np.uint8)
            lane_lines_3 = cv2.cvtColor(lane_lines, cv2.COLOR_GRAY2RGB)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(lane_lines_3, "rgb8"))
        except CvBridgeError as e:
            print(e)
        end = time.clock()
        print("Latency: " + str((end - start) * 1000) + " milliseconds.")

def main(args):
    rospy.init_node('lane_finder', anonymous=True)
    lf = lane_finder()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")

if __name__ == '__main__':
    main(sys.argv)