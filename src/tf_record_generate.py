# Author: Akash Bangera (2019)
# tfrecords generation script for CRNN
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from dataset_util import load_data, Constants
from multiprocessing import Pool


tf.enable_eager_execution()


def _bytes_feature(value):
  '''
  Description: Returns a bytes_list from a string / byte.
  '''
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_example(image, label):
    '''
    Description: Create a dictionary with features that may be relevant.
    '''
    try:
        image = cv2.imencode('.jpg', image)[1].tostring()
        label = str.encode(label)

        feature = {
          'label': _bytes_feature(label),
          'image_raw': _bytes_feature(image),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))
    except:
        pass


def main(argv=None):
    try:
        p = Pool(5) # process 5 images simultaneously
        images_train, image_labels_train, images_val, image_labels_val = load_data(Constants.WORDS_FILE, p)
        
        # Write the raw image files to images.tfrecords.
        # First, process the two images into tf.Example messages.
        # Then, write to a .tfrecords file.
        record_file = Constants.TRAIN_TFRECORD
        with tf.io.TFRecordWriter(record_file) as writer:
          for i in range(len(images_train)):
            tf_example = image_example(images_train[i], image_labels_train[i])
            writer.write(tf_example.SerializeToString())
        print("Successfully generated "+record_file)
            
        record_file = Constants.VAL_TFRECORDS
        with tf.io.TFRecordWriter(record_file) as writer:
          for i in range(len(images_val)):
            tf_example = image_example(images_val[i], image_labels_val[i])
            writer.write(tf_example.SerializeToString())
        print("Successfully generated "+record_file)

    except:
        raise

if __name__ == '__main__':
    tf.app.run()
