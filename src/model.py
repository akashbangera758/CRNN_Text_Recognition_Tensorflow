# Author: Akash Bangera (2019)
# model code for CRNN
import os
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Activation, LSTM, Bidirectional
import editdistance
import cv2
from dataset_util import Constants
from vis_util import bcolors


class CRNN(tf.keras.Model):
	'''
	Description: Class to create the CRNN model.
	'''
	def __init__(self, len_charList):
	    super(CRNN, self).__init__()
	    self.len_charList = len_charList
	    # CNN Layer 1
	    self.conv1 = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')
	    self.pool1 = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')
	    # CNN Layer 2
	    self.conv2 = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
	    self.pool2 = MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')
	    # CNN Layer 3
	    self.conv3 = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False)
	    # CNN Layer 4
	    self.conv4 = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', use_bias=False)
	    self.pool4 = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid')
	    # CNN Layer 5
	    self.conv5 = Convolution2D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)
	    self.norm5 = BatchNormalization(trainable=True, scale=True)
	    self.relu5 = Activation(activation='relu')
	    # CNN Layer 6
	    self.conv6 = Convolution2D(filters=512, kernel_size=3, strides=1, padding='same', use_bias=False)
	    self.norm6 = BatchNormalization(trainable=True, scale=True)
	    self.relu6 = Activation(activation='relu')
	    self.pool6 = MaxPooling2D(pool_size=(1,2), strides=(1,2), padding='valid')
	    # CNN Layer 7
	    self.conv7 = Convolution2D(filters=512, kernel_size=2, strides=(1,2), padding='same', activation='relu', use_bias=False)
	    # RNN Layer 1
	    self.rnn1 = Bidirectional(layer=LSTM(units=256, unit_forget_bias=True, dropout=0.5, return_sequences=True), merge_mode='concat')
	    # RNN Layer 2
	    self.rnn2 = Bidirectional(layer=LSTM(units=256, unit_forget_bias=True, dropout=0.5, return_sequences=True), merge_mode='concat')
	    # Atrous Layer
	    self.atrous_conv = Convolution2D(filters=self.len_charList+1, kernel_size=3, dilation_rate=(1,1), padding='same')
	    
	def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
	    
        conv1 = self.conv1(inputs)
	    pool1 = self.pool1(conv1)

	    conv2 = self.conv2(pool1)
	    pool2 = self.pool2(conv2)

	    conv3 = self.conv3(pool2)

	    conv4 = self.conv4(conv3)
	    pool4 = self.pool4(conv4)

	    conv5 = self.conv5(pool4)
	    norm5 = self.norm5(conv5)
	    relu5 = self.relu5(norm5)

	    conv6 = self.conv6(relu5)
	    norm6 = self.norm6(conv6)
	    relu6 = self.relu6(norm6)
	    pool6 = self.pool6(relu6)

	    conv7 = self.conv7(pool6)
	    x = tf.squeeze(conv7, axis=[2])

	    rnn1 = self.rnn1(x)
	    rnn2 = self.rnn2(rnn1)
	    x = tf.expand_dims(rnn2, axis=2)

	    atrous_conv = self.atrous_conv(x)
	    x = tf.squeeze(atrous_conv, axis=[2])

	    return x

    def initialize_hidden_state(self):
        return tf.zeros((Constants.BATCH_SIZE, Constants.REQUIRED_WIDTH, Constants.REQUIRED_HEIGHT, Constants.CHANNEL))


def ctc_loss(model_op, gText):
    '''
    Description: Function to calculate CTC loss on output of model and decode the output.

    Inputs:
    model_op: Output obtained from model.
    gText: Ground truth sparse text.

    Outputs:
    loss: Calculated CTC loss.
    decoder: Decoded output obtained using greedy decoder.
    '''
    model_op_t = tf.transpose(model_op, [1, 0, 2])
    loss = tf.reduce_mean(tf.nn.ctc_loss(labels=gText, inputs=model_op_t, sequence_length=[Constants.MAX_TEXT_LENGTH]*Constants.BATCH_SIZE))
    decoder = tf.nn.ctc_greedy_decoder(inputs=model_op_t, sequence_length=[Constants.MAX_TEXT_LENGTH]*Constants.BATCH_SIZE)

    return loss, decoder


def train_on_batch(model, ds, batch_count, charList, optimizer):
    '''
    Description: Trains the model on batch of images.

    Inputs:
    model: Instance of model to be trained.
    ds: Tensorflow dataset containing images and labels.
    batch_count: Number of batches in dataset.
    charList: A list containing all unique characters from training set.
    optimizer: Optimizer for training the model.

    Outputs:
    loss: CTC loss calculated.
    decoder: Decoded output obtained using greedy decoder.
    '''
    try:
        print("Training")
        for i in range(batch_count):
            images, labels = next(iter(ds))
            images = images.numpy()
            labels = [label.decode('utf-8') for label in labels.numpy()]
            labels = toSparse(labels, charList)

            with tf.GradientTape() as tape:
                model_op = model(images)
                loss, decoder = ctc_loss(model_op, labels)

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))

            print("Batch: "+str(i)+"/"+str(batch_count)+" Loss: "+str(loss.numpy()))

        return loss.numpy()
    
    except:
        raise


def validate_on_batch(model, ds, batch_count, charList, epoch):
    '''
    Description: Evaluates the model on batch of images.

    Inputs:
    model: Instance of model to be trained.
    ds: Tensorflow dataset containing images and labels.
    batch_count: Number of batches in dataset.
    charList: A list containing all unique characters from training set.
    epoch: Epoch count of training.

    Outputs:
    charErrorRate: Character error obtained on test set
    wordAccuracy: Word Accuracy obtained on test set
    '''
    try:
        print("Validation")
        numCharErr = 0
        numCharTotal = 0
        numWordOK = 0
        numWordTotal = 0

        for i in range(batch_count):
            images, labels = next(iter(ds))
            images = images.numpy()
            labels = [label.decode('utf-8') for label in labels.numpy()]

            model_op = model(images)
            model_op_t = tf.transpose(model_op, [1, 0, 2])
            decoder = tf.nn.ctc_greedy_decoder(inputs=model_op_t, sequence_length=[Constants.MAX_TEXT_LENGTH]*Constants.BATCH_SIZE)
            recognized = decoderOutputToText(decoder, charList)

            print("Batch: "+str(i)+"/"+str(batch_count))
            print('Ground truth -> Recognized')
            for i in range(Constants.BATCH_SIZE):
                numWordOK += 1 if labels[i] == recognized[i] else 0
                numWordTotal += 1
                dist = editdistance.eval(recognized[i], labels[i])
                numCharErr += dist
                numCharTotal += len(labels[i])
                print(bcolors.OKGREEN+'[OK]' if dist==0 else bcolors.FAIL+'[ERR:%d]' % dist,'"' + labels[i] + '"', '->', '"' + recognized[i] + '"'+bcolors.ENDC)

        # print validation result
        charErrorRate = (numCharErr / numCharTotal) * 100
        wordAccuracy = (numWordOK / numWordTotal) * 100
        print('EPOCH '+str(epoch)+': Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate, wordAccuracy))
        with open(Constants.ACCURACY_FILE, "a") as f:
            f.write('EPOCH '+str(epoch)+': Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate, wordAccuracy)+'\n')

        return images, recognized, charErrorRate, wordAccuracy

    except:
        raise


def save_model(model, epoch):
    '''
    Description: Function for saving checkpoints of the model
    '''
    checkpoint_dir = Constants.MODEL_DIR+str(epoch)
    os.mkdir(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    saver = tfe.Saver(model.variables)
    saver.save(checkpoint_prefix)


def toSparse(texts, charList):
    '''
    Description: Converts ground truth texts into sparse tensor for ctc_loss.

    Input:
    texts: String containing ground truth text.
    charList: A list containing all unique characters from training set.

    Output:
    sparseText: Sparse tensor containing the ground truth.
    '''
    indices = []
    values = []
    shape = [len(texts), 0] # last entry must be max(labelList[i])

    # go over all texts
    for (batchElement, text) in enumerate(texts):
        # text = text.decode("utf-8")
        # convert to string of label (i.e. class-ids)
        labelStr = [charList.index(c) for c in text]
        # sparse tensor must have size of max. label-string
        if len(labelStr) > shape[1]:
            shape[1] = len(labelStr)
        # put each label into sparse tensor
        for (i, label) in enumerate(labelStr):
            indices.append([batchElement, i])
            values.append(label)

    sparseText = tf.SparseTensor(indices, values, shape)
    return sparseText


def decoderOutputToText(ctcOutput, charList):
    '''
    Description: Extract texts from output of CTC decoder.

    Input:
    ctcOutput: Output obtained from model.
    charList: A list containing all unique characters from training set.

    Output:
    decodedText: A list containing all decoded strings.
    '''
    # contains string of labels for each batch element
    encodedLabelStrs = [[] for i in range(Constants.BATCH_SIZE)]

    decoded=ctcOutput[0][0]

    # go over all indices and save mapping: batch -> values
    for (idx, idx2d) in enumerate(decoded.indices):
        label = decoded.values[idx]
        batchElement = idx2d[0] # index according to [b,t]
        encodedLabelStrs[batchElement].append(label)

    # map labels to chars for all batch elements
    decodedText = [str().join([charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]
    return 	decodedText