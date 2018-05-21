import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import cv2
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))
yTrain = np.squeeze(yTrain)
yTest = np.squeeze(yTest)

x = tf.placeholder(tf.float32, [None, 3072], name="input")
y_ = tf.placeholder(tf.int64, [None])

W = tf.Variable(tf.zeros([3072, 10]))
b = tf.Variable(tf.zeros([10]))
xx = tf.reshape(x, [-1, 3072],)

y = tf.matmul(x, W) + b
y1 = tf.nn.softmax(tf.matmul(xx, W) + b, name="output")

cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_, 10), logits=y))
train_step = tf.train.AdamOptimizer(0.0000001).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def train():

    print("Loop   Train Loss   Train Acc%   Test Loss    Test Acc%")

    for i in range(2000):
        s = np.arange(xTrain.shape[0])
        np.random.shuffle(s)
        xTr = xTrain[s]
        yTr = yTrain[s]
        batch_xs = xTr[:512]
        batch_ys = yTr[:512]

        loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_loss,_ = sess.run([cross_entropy, train_step], feed_dict={x: xTrain, y_: yTrain})
        train_accuracy = sess.run(accuracy, feed_dict={x: xTrain, y_: yTrain})

        test_loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: xTest, y_: yTest})
        test_accuracy = (sess.run(accuracy, feed_dict={x: xTest, y_: yTest}))

        if i % 200 == 0:
            print('%3s' % int(i / 200 + 1 ) , '%9s' % round(train_loss, 4), '%13s' % round(train_accuracy * 100, 4),
                  '%13s' % round(test_loss, 4), '%11s' % round(test_accuracy * 100, 4))

    saver = tf.train.Saver()
    saver.save(sess, 'model/model')

def test(temp):

    filename = temp

    saver = tf.train.Saver()
    saver.restore(sess, 'model/model')

    tf.get_default_graph().as_graph_def()
    a = sess.graph.get_tensor_by_name("input:0")
    b_con = sess.graph.get_tensor_by_name("output:0")

    Image = cv2.imread(filename, 1)
    img = cv2.resize(Image, (32, 32))
    resized_image = np.array(img).reshape(1, 32, 32, 3)
    resized_image = np.reshape(resized_image, (resized_image.shape[0], -1))

    classification = sess.run(b_con, feed_dict={x: resized_image})

    max = sess.run(tf.argmax(classification, 1))
    m = max[0]
    cifar10classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print(cifar10classes[m])

if sys.argv[1] == "train":
    train()
else:
    test(sys.argv[2])
