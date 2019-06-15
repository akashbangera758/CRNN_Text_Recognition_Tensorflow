# Author: Akash Bangera (2019)
# Dataset utils code for CRNN
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import pickle
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

class Constants:
    WORDS_FILE = "../data/words.txt"
    CHARLIST_FILE = "../data/charList.txt"
    TRAIN_TFRECORD = "../data/train.tfrecords"
    VAL_TFRECORDS = "../data/val.tfrecords"
    ACCURACY_FILE = "../model/accuracy.txt"
    MODEL_DIR = "../model/saved_model/"
    TENSORBOARD_DIR = "../model/train/"
    AUTOTUNE = tf.contrib.data.AUTOTUNE
    BATCH_SIZE = 50
    MAX_TEXT_LENGTH = 32
    REQUIRED_HEIGHT = 32
    REQUIRED_WIDTH = 128
    CHANNEL = 1 # For Grayscale
    VAL_SIZE = 0.15


def load_data(filename, pool):
    '''
    Description: Loads images and labels from text file. Finds all unique characters from training data.

    Input:
    filename: Text file containing image paths and their corresponding annotations.
              It should be in the format "/path/to/image\tannotation\n".
    pool: Multiprocessing pool

    Output:
    images_train: Array containing all train images.
    image_labels_train: Array containing labels corresponding to images in all_images_train.
    images_val: Array containing all validation images.
    image_labels_val: Array containing labels corresponding to images in all_images_val.
    '''
    try:
        data = pd.read_csv(filename, sep="\t", names=["filepaths", "labels"], quoting=csv.QUOTE_NONE)
        image_paths = data["filepaths"].values.tolist()
        labels = data["labels"].values.tolist()

        image_paths, labels = shuffle(image_paths, labels, random_state=0)
        image_paths_train, image_paths_val, labels_train, labels_val = train_test_split(image_paths, labels, test_size=Constants.VAL_SIZE, random_state=42)

        # Find out all unique characters in training set
        chars = set()
        for word in labels:
            gtText = truncateLabel(' '.join(word), Constants.MAX_TEXT_LENGTH)
            chars = chars.union(set(list(gtText)))
        charList = sorted(list(chars))
        with open(Constants.CHARLIST_FILE, "wb") as fp:
            pickle.dump(charList, fp)

        image_labels_train = np.array(labels_train)
        image_labels_val = np.array(labels_val)

        images_train = pool.map(load_and_preprocess_image, image_paths_train)
        images_train = np.reshape(images_train, (-1, Constants.REQUIRED_WIDTH, Constants.REQUIRED_HEIGHT))
        images_val = pool.map(load_and_preprocess_image, image_paths_val)
        images_val = np.reshape(images_val, (-1, Constants.REQUIRED_WIDTH, Constants.REQUIRED_HEIGHT))

        return images_train, image_labels_train, images_val, image_labels_val

    except:
        raise


def _parse_image_function(example_proto):
    '''
    Description: Create a dictionary describing the features.
    '''
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


def decode_standardize_raw_image(image_raw):
    '''
    Description: Decodes raw images to get cv2 array and linearly scales it to have zero mean and unit variance.
    '''
    image = cv2.imdecode(np.frombuffer(image_raw, np.uint8), -1)
    image = tf.reshape(image, shape=[Constants.REQUIRED_WIDTH, Constants.REQUIRED_HEIGHT, Constants.CHANNEL])
    image = tf.image.per_image_standardization(image)

    return image


def read_from_tfrecord(tfrecords_file, pool):
    '''
    Description: Reads images and labels from tfrecords file.
    '''
    image_label_dataset = tf.data.TFRecordDataset(tfrecords_file)
    
    parsed_image_label_dataset = image_label_dataset.map(_parse_image_function)
    images_raw = np.array([])
    labels = np.array([])
    for image_features in parsed_image_label_dataset:
        images_raw = np.append(images_raw, image_features['image_raw'].numpy())
        labels = np.append(labels, image_features['label'].numpy().decode('utf-8'))

    images = pool.map(decode_standardize_raw_image, images_raw)
    images = tf.reshape(images, shape=[-1, Constants.REQUIRED_WIDTH, Constants.REQUIRED_HEIGHT, Constants.CHANNEL])

    return images, labels


def create_datasets(tfrecords_file):
    '''
    Creates Tensorflow dataset comprising of images and labels
    '''
    try:
        p = Pool(5) # process 5 images simultaneously
        images, labels = read_from_tfrecord(tfrecords_file, p)
        image_count = len(images.numpy())

        # Create a tensorflow dataset object containing images and labels
        ds = tf.data.Dataset.from_tensor_slices((images.numpy(), labels))

        # Setting a shuffle buffer size as large as the dataset ensures that the data is
        # completely shuffled.
        ds = ds.shuffle(buffer_size=image_count)
        ds = ds.repeat()
        ds = ds.batch(Constants.BATCH_SIZE)
        # `prefetch` lets the dataset fetch batches, in the background while the model is training.
        ds = ds.prefetch(buffer_size=Constants.AUTOTUNE)

        return ds, image_count

    except:
        raise


def truncateLabel(text, maxTextLen):
    '''
    Description: ctc_loss can't compute loss if it cannot find a mapping between text label and input labels.
                 Repeat letters cost double because of the blank symbol needing to be inserted.
                 If a too-long label is provided, ctc_loss returns an infinite gradient

    Input:
    text: Each label in list of labels.
    maxTextLen: Max Text Length to be used for training.

    Output:
    text: Returns the original text if the number of characters is less than maxTextLen.
          Else returns text upto maxTextLen.
    '''
    try:
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]

        return text

    except:
        raise


def load_and_preprocess_image(path):
    '''
    Description: Reads image from path, resizes it, and linearly scales it to have zero mean and unit variance.

    Input:
    path: Path of image to be read.

    Output:
    image: Image tensor after resize and standardization.
    '''
    try:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.zeros([Constants.REQUIRED_HEIGHT, Constants.REQUIRED_WIDTH])

        # create target image and copy sample image into it
        (wt, ht) = (Constants.REQUIRED_WIDTH, Constants.REQUIRED_HEIGHT)
        (h, w) = image.shape
        if h < Constants.REQUIRED_HEIGHT and w < Constants.REQUIRED_WIDTH:
            fx = w / wt
            fy = h / ht
            f = max(fx, fy)
            newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
            image = cv2.resize(image, newSize)
            target = np.ones([ht, wt]) * 255
            target[0:newSize[1], 0:newSize[0]] = image
        else:
            target = cv2.resize(image, (Constants.REQUIRED_WIDTH, Constants.REQUIRED_HEIGHT))

        image = cv2.transpose(target)

        return image

    except:
        raise