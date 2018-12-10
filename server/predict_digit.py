import tensorflow as tf
import math
import cv2
import numpy as np
from scipy import ndimage
import sys
import os

class DigitClassifier():
    def __init__(self):
        """
        a placeholder for our image data:
        None stands for an unspecified number of images
        784 = 28*28 pixel
        """
        self.x = tf.placeholder("float", [None, 784])

        # we need our weights for our neural net
        self.W = tf.Variable(tf.zeros([784,10]))
        # and the biases
        self.b = tf.Variable(tf.zeros([10]))

        """
        softmax provides a probability based output
        we need to multiply the image values x and the weights
        and add the biases
        (the normal procedure, explained in previous articles)
        """
        self.y = tf.nn.softmax(tf.matmul(self.x,self.W) + self.b)

        """
        y_ will be filled with the real values
        which we want to train (digits 0-9)
        for an undefined number of images
        """
        self.y_ = tf.placeholder("float", [None,10])

        """
        we use the cross_entropy function
        which we want to minimize to improve our model
        """
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))

        """
        use a learning rate of 0.01
        to minimize the cross_entropy error
        """
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)

        self.sess = tf.Session()

        # initialize all variables and run init
        self.sess.run(tf.global_variables_initializer())

    def read_training_checkpoint(self, train=False):
        checkpoint_dir = "cps/"

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found')
            exit(1)

    def predict_from_img(self, img_path):
        # read the bw image
        gray_complete = cv2.imread(img_path, 0)

        # better black and white version
        _, gray_complete = cv2.threshold(255-gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        height, width = gray_complete.shape
        side_len = min(height, width)
        start_x = width // 2
        start_y = height // 2
        diff = min(height, width) // 2
        gray = gray_complete[start_y - diff: start_y + diff, start_x - diff: start_x + diff]

        while np.sum(gray[0]) == 0:
            gray = gray[1:]

        while np.sum(gray[-1]) == 0:
            gray = gray[:-1]

        while np.sum(gray[:, 0]) == 0:
            gray = np.delete(gray, 0, 1)

        while np.sum(gray[:, -1]) == 0:
            gray = np.delete(gray, -1, 1)

        rows, cols = gray.shape

        if rows > cols:
            factor = 20.0 / rows
            rows = 20
            cols = int(round(cols * factor))
            gray = cv2.resize(gray, (cols, rows))
        else:
            factor = 20.0 / cols
            cols = 20
            rows = int(round(rows * factor))
            gray = cv2.resize(gray, (cols, rows))

        colsPadding = int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0))
        rowsPadding = int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0))
        gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

        shiftx, shifty = getBestShift(gray)
        shifted = shift(gray, shiftx, shifty)
        gray = shifted

        flatten = gray.flatten() / 255.0
        prediction = tf.argmax(self.y, 1)

        img_in = [flatten]
        vals_in = [np.zeros((10))]
        pred = self.sess.run(prediction, feed_dict={self.x: img_in, self.y_: vals_in})

        return pred[0]

def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    return shiftx, shifty

def shift(img, sx, sy):
    rows,cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted

def squareCropAbout(x, y, side_len):
    return
