# Author: Akash Bangera (2019)
# training script for CRNN
import os
import tensorflow as tf
import numpy as np
import pickle
from dataset_util import create_datasets, Constants
from model import CRNN, train_on_batch, validate_on_batch, save_model
from vis_util import write_to_tensorboard

tf.enable_eager_execution()


def main(argv=None):
	if not os.path.exists(Constants.MODEL_DIR):
	    os.makedirs(Constants.MODEL_DIR)
	if not os.path.exists(Constants.TENSORBOARD_DIR):
	    os.makedirs(Constants.TENSORBOARD_DIR)

	with open(Constants.CHARLIST_FILE, "rb") as fp:
		charList = pickle.load(fp)
	lenCharList = len(charList)

	with tf.device("CPU:0"):
		train_ds, train_image_count = create_datasets(Constants.TRAIN_TFRECORD)
		val_ds, val_image_count = create_datasets(Constants.VAL_TFRECORDS)

	train_batches = int(np.floor(train_image_count/Constants.BATCH_SIZE))
	val_batches = int(np.floor(val_image_count/Constants.BATCH_SIZE))

	model = CRNN(lenCharList)

	global_step_op = tf.Variable(0)
	starter_learning_rate = 0.1
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_op, decay_steps=10000, decay_rate=0.1, staircase=False)
	optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

	epoch = 1
	summary_writer = tf.contrib.summary.create_file_writer(Constants.TENSORBOARD_DIR, flush_millis=10000)
	with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
		while True:
			print("Epoch "+str(epoch))
			loss = train_on_batch(model, train_ds, train_batches, charList, optimizer)
			images, recognized, charErrorRate, wordAccuracy = validate_on_batch(model, val_ds, val_batches, charList, epoch)
			if charErrorRate < 15:
				save_model(model, epoch)
			write_to_tensorboard(epoch, images, recognized, loss, charErrorRate, wordAccuracy)
			epoch += 1


if __name__ == '__main__':
	tf.app.run()
