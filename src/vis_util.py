# Author: Akash Bangera (2019)
# Tensorboard utils code for CRNN
import tensorflow as tf
import cv2
from dataset_util import Constants


class tf_image:
    no_of_images = 10
    height = Constants.REQUIRED_HEIGHT + 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,45)
    fontScale = 0.5
    fontColor = (255,255,255)
    lineType = 2


class bcolors:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def write_to_tensorboard(epoch, image_batch, recognized, loss, charErrorRate, wordAccuracy):
	'''
	Description: Function for writing loss, character error rate, word accuracy and predicted text with image on tensorboard.
	'''
	tf.contrib.summary.scalar("loss", loss, step=epoch)
	tf.contrib.summary.scalar("character error rate", charErrorRate, step=epoch)
	tf.contrib.summary.scalar("word accuracy", wordAccuracy, step=epoch)
	tf_images = []
	images = image_batch
	images = tf.reshape(images, [-1, Constants.REQUIRED_WIDTH, Constants.REQUIRED_HEIGHT, Constants.CHANNEL])
	for i in range(tf_image.no_of_images):
		img = cv2.transpose(images[i].numpy())
		paddings = tf.constant([[0, 20], [0, 0]])
		img = tf.pad(img, paddings, "CONSTANT")
		img = cv2.putText(img.numpy(), recognized[i],
					tf_image.bottomLeftCornerOfText,
					tf_image.font,
					tf_image.fontScale,
					tf_image.fontColor,
					tf_image.lineType)
		tf_images.append(img)
	tf_images = tf.reshape(tf_images, shape=[-1, tf_image.height, Constants.REQUIRED_WIDTH, Constants.CHANNEL])
	tf_images = tf.cast(tf_images, dtype=tf.float32)
	tf.contrib.summary.image("recognized text", tf_images, step=epoch, max_images=10)