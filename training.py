import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import np_utils
import tensorflow as tf
import math
import time
from tensorflow.python.framework import graph_util

def load_data():
    data = h5py.File("h5img//data.h5","r")
    X_data = np.array(data['X'])
    Y_data = np.array(data['Y'])
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, train_size=0.9, test_size=0.1, random_state=22)
    print(X_train.shape)
    print(X_test.shape)
    X_train = X_train/255.
    X_test = X_test/255.

    print(y_train.shape)
    print(y_test.shape)
    y_train = np_utils.to_categorical(y_train, num_classes=11)
    print(y_train.shape)
    y_test = np_utils.to_categorical(y_test, num_classes=11)
    print(y_test.shape)
    return X_train,X_test,y_train,y_test

def w_variable(shape):
    tf.set_random_seed(1)
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def b_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(z):
    return tf.nn.max_pool(z, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def random_mini_batches(X, Y, mini_batch_size=16, seed=0):
    """
    	Creates a list of random minibatches from (X, Y)
    	Arguments:
    	X -- input data, of shape (input size, number of examples)
    	Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    	mini_batch_size - size of the mini-batches, integer
    	seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    	Returns:
    	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    	"""

    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Step 3: Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def cnn_model(X_train, y_train, X_test, y_test, keep_prob, lamda, num_epochs = 450, minibatch_size = 16):
    X = tf.placeholder(tf.float32, [None, 64, 64, 3], name="input_x")
    y = tf.placeholder(tf.float32, [None, 2], name="input_y")
    kp = tf.placeholder_with_default(1.0, shape=(), name="keep_prob")
    lam = tf.placeholder(tf.float32, name="lamda")

    # 第一次卷积
    W_conv1 = w_variable([5, 5, 3, 32])
    b_conv1 = b_variable([32])
    z1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

    # 第一次池化
    maxpool1 = max_pool_2x2(z1)  # max_pool1完后maxpool1维度为[?,32,32,32]

    # 第二次卷积
    W_conv2 = w_variable([5, 5, 32, 64])
    b_conv2 = b_variable([64])
    z2 = tf.nn.relu(conv2d(maxpool1, W_conv2) + b_conv2)

    # 第二次池化
    maxpool2 = max_pool_2x2(z2)  # max_pool2,shape [?,16,16,64]

    # 全连接层
    W_fc1 = w_variable([16 * 16 * 64, 200])
    b_fc1 = b_variable([200])
    maxpool2_flat = tf.reshape(maxpool2, [-1, 16 * 16 * 64])
    z_fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W_fc1) + b_fc1)
    z_fc1_drop = tf.nn.dropout(z_fc1, keep_prob=kp)

    # softmax层
    W_fc2 = w_variable([200, 11])
    b_fc2 = b_variable([11])
    z_fc2 = tf.add(tf.matmul(z_fc1_drop, W_fc2), b_fc2, name="outlayer")
    prob = tf.nn.softmax(z_fc2, name="probability")
    # 损失计算
    regularizer = tf.contrib.layers.l2_regularizer(lam)
    regularization = regularizer(W_fc1) + regularizer(W_fc2)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z_fc2)) + regularization

    train = tf.train.AdamOptimizer().minimize(cost)
    # output_type='int32', name="predict"
    pred = tf.argmax(prob, 1, output_type="int32", name="predict")  # 输出结点名称predict方便后面保存为pb文件
    correct_prediction = tf.equal(pred, tf.argmax(y, 1, output_type='int32'))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.set_random_seed(1)  # to keep consistent results

    seed = 0

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            seed = seed+1
            epoch_cost = 0
            num_minibatched = int(X_train.shape[0]/minibatch_size)
            minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _,minibatch_cost = sess.run([train, cost], feed_dict={X: minibatch_X, y: minibatch_Y, kp: keep_prob, lam: lamda})
                epoch_cost+=minibatch_cost/num_minibatched
            if epoch % 10==0:
                print("第%i个epoch 成本:%f" % (epoch, epoch_cost))
                print(str((time.strftime('%Y-%m-%d %H:%M:%S'))))
                train_acc = accuracy.eval(feed_dict={X: X_train[:1000], y: y_train[:1000], kp: 0.8, lam: lamda})
                print("训练精度", train_acc)

        test_acc = accuracy.eval(feed_dict={X: X_test[:1000], y: y_test[:1000], lam: lamda})
        print("测试精度", test_acc)
        saver = tf.train.Saver({'W_conv1': W_conv1, 'b_conv1': b_conv1, 'W_conv2': W_conv2, 'b_conv2': b_conv2,
                                'W_fc1': W_fc1, 'b_fc1': b_fc1, 'W_fc2': W_fc2, 'b_fc2': b_fc2})
        saver.save(sess, "model//model.ckpt")
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=['predict'])
        with tf.gfile.FastGFile('model//digital_gesture.pb',mode='wb') as f:  # ’wb’中w代表写文件，b代表将数据以二进制方式写入文件。
            f.write(output_graph_def.SerializeToString())


if __name__ == "__main__":
    print("载入数据集: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    X_train, X_test, y_train, y_test = load_data()
    #print("开始训练: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))
    #cnn_model(X_train, y_train, X_test, y_test, 0.8, 1e-4, num_epochs=200, minibatch_size=16)
    #print("训练结束: " + str((time.strftime('%Y-%m-%d %H:%M:%S'))))