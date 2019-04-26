import os
import sys
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.preprocessing import label_binarize
np.set_printoptions(threshold=np.inf)
# tf.enable_eager_execution()

# orig_stdout = sys.stdout
# f = open('MC-DCNN_results_record.txt', 'w')

TRAINING_STEP = 2000
# 配置神经网络参数
INPUT_NODE = 91
OUTPUT_NODE = 5

NUM_CHANNELS = 6
NUM_LABELS = 5
BATCH_SIZE = 300
# 第一层卷积层尺寸和深度
CONV1_DEEP = 4
CONV1_SIZE = 5
# 第二层卷基层尺寸和深度
CONV2_DEEP = 1
CONV2_SIZE = 5
# 全连接层节点个数
FC1_SIZE = 40  # 500 -> 100
FC2_SIZE = 20  # 128 -> 20

LEARNING_RATE_BASE = 0.00003  # #
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99  # (momentum)
display_step = 5
test_batch = 10000

# load data
tf.reset_default_graph()
home_dir = os.path.expanduser('~')
train_file = os.path.join(
    home_dir,
    'data_pool/waterfall_data/pretrain_result/yunjie/mississipi',
    '0401_0630_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz'
)
train_data = np.load(train_file)
x_train = train_data['features']  # (-1, 546)
x_train = x_train.reshape((-1, NUM_CHANNELS, INPUT_NODE))  # (-1, 6, 91)
y_train = train_data['labels']  # (-1, 1)
y_train[y_train == np.int64(6)] = np.int64(4)
decay_step = int(len(x_train)/BATCH_SIZE)

test_file = os.path.join(
    home_dir,
    'data_pool/waterfall_data/pretrain_result/yunjie/mississipi',
    '0401_0630_17_1_CoSoOtCoRi_L_REG_TEST_17.npz'
)
test_data = np.load(test_file)
x_test = test_data['features']
x_test = x_test.reshape((-1, NUM_CHANNELS, INPUT_NODE))
y_test = test_data['labels']
y_test[y_test == np.int64(6)] = np.int64(4)

print("TRAINING TOTAL {}".format(x_train.shape))
print("TESTING TOTAL {}".format(x_test.shape))


def iterate_minibatches(x, y, batchsize, shuffle=False):
    """
    create mini-batch data sets randomly
    :param x:
    :param y:
    :param batchsize:
    :param shuffle:
    :return:
    """
    assert len(x) == len(y)
    # if shuffle:
    #     indices = np.arange(len(x))
    #     np.random.shuffle(indices)
    for start_idx in range(0, len(x) - batchsize + 1, batchsize):
        if shuffle:
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            idx = indices[start_idx:start_idx + batchsize]
        else:
            idx = slice(start_idx, start_idx + batchsize)
        yield x[idx], y[idx]


# 定义前向传播过程
def build_mcdcnn(input_var=[None]*NUM_CHANNELS, regu=[]):
    """
    create a 6-channel DCNN, which contains 2 convolution layers, 2 pooling layers and 2 MLP layers per channel
    :param input_var:
    :param regu:
    :return:
    """
    final_res = {}
    out_sig = {}
    out_logit = {}
    for i in range(NUM_CHANNELS):
        input_tensor = input_var[i]
        print('input: ', input_tensor.get_shape().as_list())
        # input_tensor: 3-d, [batch_size, in-channel=1, in-width=91]
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable(
                'weight1'+str(i), [CONV1_SIZE, 1, CONV1_DEEP],
                initializer=tf.glorot_uniform_initializer()
            )
            conv1_biases = tf.get_variable(
                'bias1'+str(i), [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
            )
            # 定义卷积层
            conv1 = tf.nn.conv1d(
                input_tensor, conv1_weights, stride=1, padding='VALID',
                data_format='NCW', name='conv1'+str(i)
            )
            # print(tf.shape(conv1))
            # print('conv1: ', conv1.get_shape().as_list())
            # 定义激活层
            tanh1 = tf.nn.tanh(
                tf.layers.batch_normalization(conv1, name='norm1'+str(i)),
                name='tanh1' + str(i)
            )
            # print(tf.shape(tanh1))
            # print('tanh1: ', tanh1.get_shape().as_list())

        with tf.variable_scope('layer2-pool1'):
            pool1 = tf.layers.max_pooling1d(
                tanh1, pool_size=2, strides=2, padding='SAME',
                data_format='channels_first', name='pool1'+str(i)
            )
            # print(tf.shape(pool1))
            # print('pool1: ', pool1.get_shape().as_list())

        with tf.variable_scope('layer3-conv2'):
            conv2_weights = tf.get_variable(
                'weight2'+str(i), [CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                initializer=tf.glorot_uniform_initializer()
            )
            conv2_biases = tf.get_variable(
                'bias2'+str(i), [CONV2_DEEP], initializer=tf.constant_initializer(0.0)
            )
            # 定义卷积层
            conv2 = tf.nn.conv1d(
                pool1, conv2_weights, stride=1, padding='VALID',
                data_format='NCW', name='conv2'+str(i)
            )
            # print(tf.shape(conv2))
            print('conv2: ', conv2.get_shape().as_list())
            # 定义激活层
            tanh2 = tf.nn.tanh(
                tf.layers.batch_normalization(conv2, name='norm2'+str(i)),
                name='tanh2'+str(i)
            )
            # print(tf.shape(tanh2))
            print('tanh2: ', tanh2.get_shape().as_list())

        with tf.variable_scope('layer4-pool2'):
            pool2 = tf.layers.max_pooling1d(
                tanh2, pool_size=2, strides=2, padding='SAME',
                data_format='channels_first', name='pool2'+str(i)
            )
            # print(tf.shape(pool2))
            print('pool2: ', pool2.get_shape().as_list())

        pool2_shape = pool2.get_shape().as_list()
        nodes = pool2_shape[1] * pool2_shape[2]  # == 20*4=80
        cnn_output = tf.reshape(pool2, [pool2_shape[0], nodes])
        print('cnn output: ', cnn_output.get_shape().as_list())
        with tf.variable_scope('layer5-mlp1'):
            fc1_weights = tf.get_variable(
                'weight3'+str(i), [nodes, FC1_SIZE],
                initializer=tf.glorot_uniform_initializer()
            )
            if regu is not None:
                tf.add_to_collection('losses1'+str(i), regularizer(fc1_weights))
            fc1_biases = tf.get_variable(
                'bias3'+str(i), [FC1_SIZE],
                initializer=tf.constant_initializer(0.1)
            )
            fc1 = tf.nn.tanh(
                tf.layers.batch_normalization(
                    tf.matmul(cnn_output, fc1_weights) + fc1_biases,
                    name='mlp1-norm'+str(i)
                ),
                name='mlp-tanh1'+str(i)
            )
            print('mlpl1: ', fc1.get_shape().as_list())

        with tf.variable_scope('layer6-mlp2'):
            fc2_weights = tf.get_variable(
                'weight4'+str(i), [FC1_SIZE, FC2_SIZE],
                initializer=tf.glorot_uniform_initializer()
            )
            if regu is not None:
                tf.add_to_collection('losses2'+str(i), regularizer(fc2_weights))
            fc2_biases = tf.get_variable(
                'bias4'+str(i), [FC2_SIZE],
                initializer=tf.constant_initializer(0.1)
            )
            fc2 = tf.nn.tanh(
                tf.layers.batch_normalization(
                    tf.matmul(fc1, fc2_weights) + fc2_biases,
                    name='mlp2-norm'+str(i)
                ),
                name='mlp-tanh2'+str(i)
            )
            print('mlpl2: ', fc2.get_shape().as_list())

        with tf.variable_scope('layer7-mlp3'):
            fc3_weights = tf.get_variable(
                'weight5'+str(i), [FC2_SIZE, OUTPUT_NODE],
                initializer=tf.glorot_uniform_initializer()
            )
            if regu is not None:
                tf.add_to_collection('losses3'+str(i), regularizer(fc3_weights))
            fc3_biases = tf.get_variable(
                'bias5'+str(i), [OUTPUT_NODE],
                initializer=tf.constant_initializer(0.1)
            )
            out = tf.layers.batch_normalization(
                tf.matmul(fc2, fc3_weights) + fc3_biases,
                name='mlp3-norm'+str(i)
            )  # (200,5)
            print('mlpl3: ', out.get_shape().as_list())
            out_sig[i] = tf.nn.sigmoid(out, name='sigm'+str(i))
            # out_sig[i] = tf.reshape(out_s, [1, -1, OUTPUT_NODE])  # (1, 200, 5)
            out_logit[i] = tf.nn.softmax(out, name='logit'+str(i))  # (1, 200, 5)
            # out_logit[i] = tf.reshape(out_l, [1, -1, OUTPUT_NODE])
            final_res[i] = tf.argmax(out_logit[i], 1)  # (200,)
            final_res[i] = tf.reshape(final_res[i], [1, -1])  # (1, 200)
    full_network = tf.concat(list(final_res.values()), axis=0, name='concat1')  # (6, 200)
    # full_sig = tf.concat(list(out_sig.values()), axis=0, name='concat2')  # (6,200,5)
    # full_logit = tf.concat(list(out_logit.values()), axis=0, name='concat3')  # (6,200,5)
    full_out = tf.reshape(full_network, [-1, NUM_CHANNELS])  # (200, 6)
    return out_sig, out_sig, full_out


x = tf.placeholder(tf.float32, shape=(NUM_CHANNELS, BATCH_SIZE, 1, INPUT_NODE), name='x-input')
y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_NODE), name='y-input')
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
global_step = tf.Variable(0, trainable=False)
# tf.executing_eagerly()
sig_y, logit_y, full_y = build_mcdcnn(x)
# full_y = tf.gather(full_y, tf.constant(list(range(BATCH_SIZE))))
# pred_y = tf.map_fn(lambda k: Counter(k).most_common(1), elems=full_y, dtype=tf.float32)
# pred_y = []

# print(pred_y.get_shape().as_list())  # (200, 1)
pred_y_tag = tf.identity(full_y, 'predict_results')
all_band_cost = []
for i in range(NUM_CHANNELS):
    band_sig_y = sig_y[i]
    band_cost = (y * tf.log(band_sig_y)) + (1-y) * tf.log(1-band_sig_y)
    all_band_cost.append(band_cost)
print(np.array(all_band_cost).shape)
cost_entropy_mean = -tf.reduce_mean(all_band_cost)
# accuracy
# correct_pred = tf.equal(pred_y, tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# confusion matrix
# confusion_mat = tf.confusion_matrix(
#     pred_y, tf.argmax(y, 1), NUM_LABELS
# )
# conf_mat_tag = tf.identity(confusion_mat, 'confusion_matrix')

learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    decay_step,
    LEARNING_RATE_DECAY
)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_entropy_mean, global_step=global_step)
init = tf.global_variables_initializer()

print('*******************************Start training************************************')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(init)
    start_time = time.time()
    kk = 1
    for step in range(1, TRAINING_STEP+1):
        num = 1
        step_loss = []
        step_acc = []
        for batch in iterate_minibatches(x_train, y_train, BATCH_SIZE, shuffle=True):
            inputs, label = batch
            # print('label type: ', type(label[0]))  # numpy.int64
            labels = label_binarize(label, range(OUTPUT_NODE))
            # split input into different channel
            chan_inputs = [inputs[:, i, :].reshape((-1, 1, INPUT_NODE)) for i in range(NUM_CHANNELS)]
            sess.run(
                optimizer, feed_dict={x: chan_inputs, y: labels}
            )
            if step % display_step == 0 or step == 1:
                batch_loss, batch_ys = sess.run(
                    [cost_entropy_mean, full_y],
                    feed_dict={x: chan_inputs, y: labels}
                )
                pred_y = []
                for ys in batch_ys:
                    counts = Counter(ys)
                    yy = counts.most_common(1)[0][0]
                    pred_y.append(yy)
                pred_y = np.array(pred_y, dtype=type(label[0]))
                print(pred_y[:10])
                print(label[:10])
                error = len(np.where(pred_y != label)[0])
                batch_acc = 1-(error/float(BATCH_SIZE))
                step_loss.append(batch_loss)
                step_acc.append(batch_acc)
                ave_loss = sum(step_loss)/float(len(step_loss))
                ave_acc = sum(step_acc)/float(len(step_acc))
                print('STEP: {} << {} '.format(step, num))
                print('Minibatch loss: {:.4f}'.format(batch_loss))
                print('Minibatch training accuracy: {:.4f}'.format(batch_acc))
                print('average batches loss: {:.4f}'.format(ave_loss))
                print('average batches acc: {:.4f}'.format(ave_acc))
                num += 1
        # kk = 1
        if step % 10 == 0:
            # 每100轮训练，进行一次数据测试
            train_time = time.time()
            # sys.stdout = f  # 开始记录print内容
            print('RECORD TIME: ', time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time())))
            print('================================Start testing=================================')
            ii = 1
            test_loss = []
            test_acc = []
            for batch in iterate_minibatches(x_test, y_test, BATCH_SIZE, shuffle=False):
                part_x, part_y = batch
                part_ys = label_binarize(part_y, range(OUTPUT_NODE))
                part_x_chan = [part_x[:, k, :].reshape((-1, 1, INPUT_NODE)) for k in range(NUM_CHANNELS)]
                test_pred = sess.run(
                    full_y, feed_dict={x: part_x_chan, y: part_ys}
                )
                test_res = []
                for ys in test_pred:
                    counts = Counter(ys)
                    yy = counts.most_common(1)[0][0]
                    test_res.append(yy)

                pred_y = np.array(test_res, dtype=type(label[0]))
                print('part_y: ', part_y.shape)
                print(part_y[:20])
                print('pred_y: ', pred_y.shape)
                print(pred_y[:20])
                error = len(np.where(pred_y != part_y)[0])
                acc = 1 - (error / float(BATCH_SIZE))
                # res_file = os.path.join(
                #     home_dir,
                #     'data_pool/U-TMP/excersize/ML_model/MC-DCNN',
                #     '2017_mississipi_CSCRO_630_' + str(ii) + '.npz'
                # )
                # np.savez(res_file, test_res.eval())
                test_acc.append(acc)
                test_conf = confusion_matrix(part_y, pred_y)
                print('test batch: ', ii)
                ii += 1
                print('test acc: {:.6f}'.format(acc))
                print('test confusion matrix:')
                print(test_conf)
            final_acc = sum(test_acc)/float(len(test_acc))
            print('model after {} training steps.'.format(step))
            print('TEST AVERAGE ACCURACY: ', final_acc)
            print('TRAINING TOTAL {}'.format(x_train.shape))
            print('TESTING TOTAL {}'.format(x_test.shape))
            print('Training time: {}'.format(train_time-start_time))
            print('Testing time: {}'.format(time.time()-train_time))
            # input('press Enter')
    # sys.stdout = orig_stdout
    # f.close()
            # save model
            if final_acc > 0.8:
                print('Saving models ...')
                saver = tf.train.Saver()
                model_dir = os.path.join(
                        home_dir,
                        'data_pool/U-TMP/excersize/ML_model/MC-DCNN',
                        'model_dir1008_RF_'+str(kk)
                    )
                kk += 1
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                os.makedirs(model_dir)
                saver.save(sess, os.path.join(model_dir, "MultiChannel-DCNN"))
