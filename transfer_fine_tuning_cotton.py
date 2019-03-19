"""
RNN model for crop classification: cotton, 3; other, 0
CAUTION: For transfer fine-tuning learning, the label of positive sample should be same as original model !!!
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

save_model_dir = "./model_dir/cotton_2018_"
# load_model_dir = "../model_dir/cotton/china_cotton"
load_model_dir = "../model_dir/cotton/final"

# NOW here we define traning and model parameters
# Training Parameters
learning_rate = 0.00001	# 0.0005 -> 0.0003
training_steps = 3000
batch_size = 256

# Network Parameters
num_input = 6
timesteps = 183
num_classes = 2
dropout = 0.4	# 0.25 -> 0.4
display_step = 10

print("Initializing data")

# get training and testing data
data = np.load(
    join(
        os.path.expanduser("~"),
        "data_pool/U-TMP/excersize/point_extractor/preprocessed_points/pretrain/North_XJ",
        "0401_0930_17_1_OtOtOtCo_L_REG_TRAIN_18.npz",
    )
)
data_feat = data["features"]
data_lab = data["labels"]

data_lab[data_lab != 3] = 1  # other = 1
data_lab[data_lab == 3] = 0  # cotton = 0

idx_otc = np.arange(79378)
idx_ot = np.arange(79378, 143925)
idx_co = np.arange(143925, 197636)
idx_ots = np.arange(197636, 238295)
# random.shuffle(idx_ots)
# random.shuffle(idx_ot)
# random.shuffle(idx_otc)
# random.shuffle(idx_co)

train_idx = list(idx_otc[: int(len(idx_otc)*0.7)]) + list(idx_ot[: int(len(idx_ot)*0.7)]) + list(idx_co[: int(len(idx_co)*0.7)]) + list(idx_ots[: int(len(idx_ots)*0.7)])
# test_idx = list(idx_otc[int(len(idx_otc)*0.7):]) + list(idx_ot[int(len(idx_ot)*0.7):]) + list(idx_co[int(len(idx_co)*0.7):]) + list(idx_ots[int(len(idx_ots)*0.7):])
test_idx = np.delete(np.arange(len(data_lab)), train_idx)

# train_idx = random.sample(range(len(data_lab)), int(len(data_lab) * 0.7))
# test_idx = np.delete(np.arange(len(data_lab)), train_idx)

random.shuffle(train_idx)
features_train = data_feat[train_idx, :]
labels_train = data_lab[train_idx]

random.shuffle(test_idx)
features_test = data_feat[test_idx, :]
labels_test = data_lab[test_idx]

# save test data set to npz file
np.savez(join(save_model_dir, "test_sample.npz"), features=features_test, labels=labels_test, ID=test_idx)

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

# run training and testing experiment
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True

with tf.Session(config=conf) as sess:
        
    # restore saved model; get operations, tensors
    saver = tf.train.import_meta_graph(load_model_dir + "/fucking_model.meta")
    saver.restore(sess, tf.train.latest_checkpoint(load_model_dir))
    graph = tf.get_default_graph()
    
    for gg in graph.get_operations():
        print(gg.values())

    train_op = graph.get_operation_by_name("train_op")
    loss_op = graph.get_tensor_by_name("loss_op:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    correct_pred = graph.get_tensor_by_name("correct_prediction:0")
    conf_matrix = graph.get_tensor_by_name("confusion_matrix/SparseTensorDenseAdd:0")

    X = graph.get_tensor_by_name("input_x:0")
    Y = graph.get_tensor_by_name("input_y:0")
    DROP_PROB = graph.get_tensor_by_name("dropout_prob:0")
    LEARN_RATE = graph.get_tensor_by_name("learning_rate:0")

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
                    feed_dict={X: batch_x, Y: batch_y, DROP_PROB: 0.0, LEARN_RATE: learning_rate},
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

            pre = conf_mm[0, 0]/float(conf_mm[0, 0] + conf_mm[0, 1])
            recall = conf_mm[0, 0]/float(conf_mm[0, 0] + conf_mm[1, 0])
            f1 = 2 * pre * recall/float(pre + recall)
            f = open(join(save_model_dir, "accuracy_loss.txt"), "a")
            f.write("STEP " + str(step) + "\n")
            f.write("TRAIN ACC " + str(batch_acc_av) + "\n")
            f.write("TRAIN LOSS " + str(batch_loss_av) + "\n")
            f.write("TEST ACC " + str(t_acc) + "\n")
            f.write("TEST LOSS " + str(t_loss) + "\n")
            f.write("CONFUSION MATRIX " + "\n")
            f.write(str(conf_mm) + "\n")
            f.write('PRECISION ' + str(pre) + '\n')
            f.write('RECALL ' + str(recall) + '\n')
            f.write('F1-SCORE ' + str(f1) + '\n\n')
            f.close()

            # save predict correct and wrong data
            np.savez(join(save_model_dir, "predict_result_"+str(step)+".npz"), test_correct)

            saver = tf.train.Saver()
            # shutil.rmtree(model_dir)
            # os.makedirs(model_dir)
            saver.save(
                sess,
                join(save_model_dir, "fucking_model"),
                global_step=step,
                write_meta_graph=True,
            )
