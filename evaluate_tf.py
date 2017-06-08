import tensorflow as tf
import cv2
from autoseg_backend import BackendHandler

num_classes = 34
data_dir = '/cityscapes_1024/'

sess = tf.Session()
saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

backend = BackendHandler(data_dir=data_dir, num_classes=num_classes, visualize_while_training=visualize_while_training)

for batch in backend.generateData(batch_size=3, validating=True):
    predictions = main.eval(feed_dict={
        input_1: batch})
    for i in range(len(predictions)):
        ID = getID()
        cv2.imshow('Image', x[i])
        cv2.moveWindow('Image', 10, 10)
        cv2.imshow('Ground Truth', makeLabelPretty( oneHotToLabel(y[i]) ))
        cv2.moveWindow('Ground Truth', 850, 10)
        cv2.imshow('Model Output', makeLabelPretty( oneHotToLabel(predictions[i]) ))
        cv2.moveWindow('Model Output', 850, 500)
        cv2.waitKey(5000)