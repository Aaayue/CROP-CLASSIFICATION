"""
RNN model for crop classification
"""
# -*- coding: utf-8 -*-
# test should be on python 3, tensofflow should use GPU backend

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from os.path import join

# import shutil

model_dir = "./model_dir/CSOR_addother2"

# import tf.python.client import device_lib
# flags=tf.flags
# logging=tf.logging
# FLAGS = flags.FLAGS

# NOW here we define traning and model parameters
# Training Parameters
learning_rate = 0.01
training_steps = 2000
batch_size = 256

# Network Parameters
num_input = 6
timesteps = 183
num_classes = 4
dropout = 0.25  # 0.4 -> 0.25
num_hidden = 192    # 320 -> 192
num_layers = 2
display_step = 20

print("Initializing data")
data = np.load(
    join(
        os.path.expanduser('~'),
        "data_pool/waterfall_data/pretrain_result/china",
        "0401_0930_17_1_CoSoOtRi_L_REG_TRAIN_18.npz"
    )
)
data_feat = data["features"]
data_lab = data["labels"]
data_lab[data_lab == 3] = 2  # rice 3 -> 2
data_lab[data_lab == 6] = 3  # other 6 -> 3

# separate data into train sat and test set
l0 = len(np.where(data_lab == 0)[0])
l1 = len(np.where(data_lab == 1)[0])
l3 = len(np.where(data_lab == 2)[0])
l6 = len(np.where(data_lab == 3)[0])
tmp1 = list(range(int(0.8*l6)))
tmp2 = list(range(l6, l6+int(0.8*l0)))
tmp3 = list(range((l0+l6), l0+l6+int(0.8*l3)))
tmp4 = list(range((l0+l6+l3), l0+l6+l3+int(0.8*l1)))
tmp1.extend(tmp2)
tmp3.extend(tmp4)
tmp1.extend(tmp3)
train_idx = tmp1
# train_idx = random.sample(range(len(data_lab)), int(len(data_lab) * 0.8))
print(train_idx[:20])
random.shuffle(train_idx)
print(train_idx[:20])
features_train = data_feat[train_idx, :]
labels_train = data_lab[train_idx]

test_idx = np.delete(range(len(data_lab)), train_idx)
random.shuffle(test_idx)
features_test = data_feat[test_idx, :]
labels_test = data_lab[test_idx]


# convert nan to 0, get rid of nan
features_train = np.nan_to_num(features_train)
features_test = np.nan_to_num(features_test)

# reshape training testing data into matrix (variable as row)
features_train = features_train.reshape((-1, num_input, timesteps))
features_train = features_train.transpose(0, 2, 1)
labels_train = tf.one_hot(labels_train, num_classes)  # one hot

features_test = features_test.reshape((-1, num_input, timesteps))
features_test = features_test.transpose(0, 2, 1)
labels_test = tf.one_hot(labels_test, num_classes)  # one hot

print("TRAINING TOTAL %d" % features_train.shape[0])
print("TESTING TOTAL %d" % features_test.shape[0])

# now create computational graph
num_batch = int(features_train.shape[0] / batch_size)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input], name="input_x")
Y = tf.placeholder("float", [None, num_classes], name="input_y")
DROP_PROB = tf.placeholder(tf.float32, name="dropout_prob")
LEARN_RATE = tf.placeholder(tf.float32, name="learning_rate")

# Define weights
weights = {"out": tf.Variable(tf.random_normal([num_hidden, num_classes]), name="weights_out")}
biases = {"out": tf.Variable(tf.random_normal([num_classes]), name="biases_out")}


def RNN(x, drop, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    # here we use static_rnn which requires inputs to be a list of tensors
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.LSTMCell(num_hidden, use_peepholes=True)
    lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=(1 - drop))
    # lstm_cell=cudnn_rnn.CudnnLSTM(num_layers=1,num_units=num_hidden,dropout=(1-dropout))
    # Get lstm cell output
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights["out"]) + biases["out"]


logits = RNN(X, DROP_PROB, weights, biases)
prediction = tf.nn.softmax(logits, name="prediction")
prediction_tag = tf.identity(prediction, name="predict_the_fuck")

# Define loss and optimizer
loss_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y),
    name="loss_op"
)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdagradOptimizer(
    learning_rate=learning_rate, initial_accumulator_value=0.1,
    name="optimizer_op"
)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, name="train_op")

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1), name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

# define confusion matrix
conf_matrix = tf.confusion_matrix(
    tf.argmax(Y, 1), tf.argmax(prediction, 1), num_classes,
    name="confusion_matrix"
)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# run training and testing experiment
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True

with tf.Session(config=conf) as sess:
    # debug
    # sess=tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan",tf_debug.has_inf_or_nan)

    # Run the initializer
    sess.run(init)
    labels_train = sess.run(labels_train)

    for step in range(1, training_steps + 1):

        # random shuffle data for each epoch for better training
        rand_array = np.arange(features_train.shape[0])
        np.random.shuffle(rand_array)
        features_train = features_train[rand_array]
        labels_train = labels_train[rand_array]

        for b in range(num_batch + 1):
            batch_x, batch_y = (
                features_train[b * batch_size : (b + 1) * batch_size],  # noqa: E203
                labels_train[b * batch_size : (b + 1) * batch_size],  # noqa: E203
            )

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, DROP_PROB: dropout, LEARN_RATE: learning_rate})

        if step % display_step == 0 or step == 1:
            batch_loss = []
            batch_acc = []
            # Calculate batch loss and accuracy
            for b in range(num_batch):
                batch_x, batch_y = (
                    features_train[b * batch_size : (b + 1) * batch_size],  # noqa: E203
                    labels_train[b * batch_size : (b + 1) * batch_size],  # noqa: E203
                )
                loss, acc = sess.run(
                    [loss_op, accuracy],
                    feed_dict={X: batch_x, Y: batch_y, DROP_PROB: 0.0, LEARN_RATE: learning_rate},
                )
                batch_loss.append(loss)
                batch_acc.append(acc)
            batch_loss_av = sum(batch_loss) / float(len(batch_loss))
            batch_acc_av = sum(batch_acc) / float(len(batch_acc))
            print(
                "Step "
                + str(step)
                + ", Minibatch Loss= "
                + "{:.4f}".format(batch_loss_av)
                + ", Training Accuracy= "
                + "{:.3f}".format(batch_acc_av)
            )

            print("Optimization Finished!")

            # Calculate accuracy for testing data
            test_data = features_test
            test_label = sess.run(labels_test)

            n = 11
            size = int(test_data.shape[0] / n)
            test_acc = []
            test_loss = []
            test_correct =[]
            for b in range(n + 1):
                batch_x, batch_y = (
                    test_data[b * size : (b + 1) * size],  # noqa :E203
                    test_label[b * size : (b + 1) * size],  # noqa :E203
                )
                loss, acc, correct, conf_m = sess.run(
                    [loss_op, accuracy, correct_pred, conf_matrix],
                    feed_dict={X: batch_x, Y: batch_y, DROP_PROB: 0.0, LEARN_RATE: learning_rate, LEARN_RATE: learning_rate},
                )
                test_acc.append(acc)
                test_loss.append(loss)
                test_correct.extend(correct)

                if b == 0:
                   conf_mm = conf_m
                else:
                   conf_mm = conf_mm + conf_m

            t_acc = sum(test_acc) / float(len(test_acc))
            t_loss = sum(test_loss) / float(len(test_loss))

            print("Testing Accuracy: %f" % t_acc)
            print("Testing Loss: %f" % t_loss)
            print("Confusion Matrix: ")
            print(conf_mm)

            PRE = []
            REC = []
            F1 = []
            for i in range(num_classes):
                pre = conf_mm[i, i]/float(np.sum(conf_mm, axis=0)[i])
                recall = conf_mm[i, i]/float(np.sum(conf_mm, axis=1)[i])
                f1 = 2 * pre * recall/float(pre + recall)
                PRE.append(pre)
                REC.append(recall)
                F1.append(f1)

            print("Testing Precision: ", PRE)
            print("Testing Recall: ", REC)
            print("Testing F1 score: ", F1)

            f = open(join(model_dir, "accuracy_loss.txt"), "a")
            f.write("STEP " + str(step) + "\n")
            f.write("TRAIN ACC " + str(batch_acc_av) + "\n")
            f.write("TRAIN LOSS " + str(batch_loss_av) + "\n")
            f.write("TEST ACC " + str(t_acc) + "\n")
            f.write("TEST LOSS " + str(t_loss) + "\n")
            f.write("CONFUSION MATRIX " + "\n")
            f.write(str(conf_mm) + "\n\n")
            f.write('PRECISION ' + str(PRE) + '\n')
            f.write('RECALL ' + str(REC) + '\n')
            f.write('F1-SCORE ' + str(F1) + '\n\n')
            f.close()

            # save predict correct and wrong data
            np.savez(join(model_dir, "predict_result_"+str(step)+".npz"), test_correct)

            saver = tf.train.Saver()
            # shutil.rmtree(model_dir)
            # os.makedirs(model_dir)
            saver.save(
                sess,
                join(model_dir, "fucking_model"),
                global_step=step,
                write_meta_graph=True,
            )
