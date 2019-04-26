"""
labels=[
    "Corn": 0,
    "Soybeans": 1,
    "Cotton": 2,
    "Rice": 3,
    "Other": 6
]
"""
import os
import sys
import time
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
np.set_printoptions(threshold=np.inf)

# orig_stdout = sys.stdout
# f = open('MC-DCNN_results_record.txt', 'w')

TRAINING_STEP = 2000
# 配置神经网络参数
INPUT_NODE = 183
OUTPUT_NODE = 5

NUM_CHANNELS = 6
NUM_LABELS = 5
BATCH_SIZE = 300
# 第一层卷积层尺寸和深度
CONV1_DEEP = 8
CONV1_SIZE = 5
# 第二层卷基层尺寸和深度
CONV2_DEEP = 4
CONV2_SIZE = 5
# 全连接层节点个数
FC1_SIZE = 100  # 500 -> 100
FC2_SIZE = 20  # 128 -> 20

LEARNING_RATE_BASE = 0.01  # 0.01 -> 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99  # (momentum)
display_step = 1
test_batch = 10000

# load data
tf.reset_default_graph()
home_dir = os.path.expanduser('~')
train_file = os.path.join(
    home_dir,
    'data_pool/waterfall_data/pretrain_result/yunjie-10/mississipi',
    '0401_0930_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz'
)
train_data = np.load(train_file)
x_train = train_data['features']
# len_train = len(x_train)
# idx = np.arange(len_train)
# np.random.shuffle(idx)
# iid = idx[:int(len_train*0.5)]
# x_train = x_train[iid]
x_train = x_train.reshape((-1, NUM_CHANNELS, INPUT_NODE))
y_train = train_data['labels']
y_train[y_train == np.int64(6)] = np.int64(4)
# y_train = y_train[iid]
# y_train = tf.one_hot(y_train, 5)
# decay_step = int(len(x_train)/BATCH_SIZE)
decay_step = 200

test_file = os.path.join(
    home_dir,
    'data_pool/waterfall_data/pretrain_result/yunjie-10/mississipi',
    '0401_0930_17_1_CoSoOtCoRi_L_REG_TEST_17.npz'
)
test_data = np.load(test_file)
x_test = test_data['features']
# len_test = len(x_test)
# idx2 = np.arange(len_test)
# np.random.shuffle(idx2)
# iid2 = idx2[:int(len_test*0.5)]
# x_test = x_test[iid2]
x_test = x_test.reshape((-1, NUM_CHANNELS, INPUT_NODE))
y_test = test_data['labels']
y_test[y_test == np.int64(6)] = np.int64(4)
# y_test = y_test[iid2]
# y_test = tf.one_hot(y_test, 5)
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
def build_mcdcnn(input_var=[None]*NUM_CHANNELS):
    """
    create a 6-channel DCNN, which contains 2 convolution layers, 2 pooling layers and 2 MLP layers per channel
    :param input_var:
    :return:
    """
    network = {}
    full_nodes = 0
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
            # print('conv2: ', conv2.get_shape().as_list())
            # 定义激活层
            tanh2 = tf.nn.tanh(
                tf.layers.batch_normalization(conv2, name='norm2'+str(i)),
                name='tanh2'+str(i)
            )
            # print(tf.shape(tanh2))
            # print('tanh2: ', tanh2.get_shape().as_list())

        with tf.variable_scope('layer4-pool2'):
            pool2 = tf.layers.max_pooling1d(
                tanh2, pool_size=2, strides=2, padding='SAME',
                data_format='channels_first', name='pool2'+str(i)
            )
            # print(tf.shape(pool2))
            print('pool2: ', pool2.get_shape().as_list())

        pool2_shape = pool2.get_shape().as_list()
        nodes = pool2_shape[1] * pool2_shape[2]  # == 20*6=120
        network[i] = tf.reshape(pool2, [pool2_shape[0], nodes])
        full_nodes += nodes
    # 将所有通道的数据整合到一个输入层中，输入MLP
    print(network.values())
    full_network = tf.concat(list(network.values()), axis=1, name='concat')
    with tf.variable_scope('layer5-mlp1'):
        fc1_weights = tf.get_variable(
            'weight3', [full_nodes, FC1_SIZE],
            initializer=tf.glorot_uniform_initializer()
        )
        fc1_biases = tf.get_variable(
            'bias3', [FC1_SIZE],
            initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(full_network, fc1_weights) + fc1_biases)
        )

    with tf.variable_scope('layer6-mlp2'):
        fc2_weights = tf.get_variable(
            'weight3', [FC1_SIZE, FC2_SIZE],
            initializer=tf.glorot_uniform_initializer()
        )
        fc2_biases = tf.get_variable(
            'bias3', [FC2_SIZE],
            initializer=tf.constant_initializer(0.1)
        )
        fc2 = tf.nn.tanh(
            tf.layers.batch_normalization(tf.matmul(fc1, fc2_weights) + fc2_biases)
        )

    with tf.variable_scope('layer7-mlp3'):
        fc3_weights = tf.get_variable(
            'weight4', [FC2_SIZE, OUTPUT_NODE],
            initializer=tf.glorot_uniform_initializer()
        )
        fc3_biases = tf.get_variable(
            'bias4', [OUTPUT_NODE],
            initializer=tf.constant_initializer(0.1)
        )
        cnn_network = tf.nn.softmax(
            tf.layers.batch_normalization(tf.matmul(fc2, fc3_weights) + fc3_biases)
        )

    return cnn_network


x = tf.placeholder(tf.float32, shape=(NUM_CHANNELS, BATCH_SIZE, 1, INPUT_NODE), name='x-input')
y = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_NODE), name='y-input')
global_step = tf.Variable(0, trainable=False)
pred_y = build_mcdcnn(x)
pred_y_tag = tf.identity(pred_y, 'predict_results')
# cost function
cost_entropy = tf.keras.losses.categorical_crossentropy(y, pred_y)
cost_entropy_mean = tf.reduce_mean(cost_entropy)
# accuracy
correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# confusion matrix
confusion_mat = tf.confusion_matrix(
    tf.argmax(pred_y, 1), tf.argmax(y, 1), NUM_LABELS
)
conf_mat_tag = tf.identity(confusion_mat, 'confusion_matrix')

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
    # sess = sess(config=config)
    sess.run(init)
    start_time = time.time()
    kk = 1
    for step in range(1, TRAINING_STEP+1):
        num = 1
        step_loss = []
        step_acc = []
        for batch in iterate_minibatches(x_train, y_train, BATCH_SIZE, shuffle=True):
            inputs, label = batch
            label = label_binarize(label, range(OUTPUT_NODE))
            # split input into different channel
            chan_inputs = [inputs[:, i, :].reshape((-1, 1, INPUT_NODE)) for i in range(NUM_CHANNELS)]
            sess.run(
                optimizer, feed_dict={x: chan_inputs, y: label}
            )
            if step % display_step == 0 or step == 1:
                batch_loss, batch_acc = sess.run(
                    [cost_entropy_mean, accuracy],
                    feed_dict={x: chan_inputs, y: label}
                )
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
        if step % 1 == 0:
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
                part_y = label_binarize(part_y, range(OUTPUT_NODE))
                part_x_chan = [part_x[:, k, :].reshape((-1, 1, INPUT_NODE)) for k in range(NUM_CHANNELS)]
                test_pred = sess.run(
                    pred_y, feed_dict={x: part_x_chan, y: part_y}
                )
                test_res = tf.argmax(test_pred, 1)
                acc, test_conf = sess.run(
                    [accuracy, confusion_mat],
                    feed_dict={x: part_x_chan, y: part_y}
                )
                # res_file = os.path.join(
                #     home_dir,
                #     'data_pool/U-TMP/excersize/ML_model/MC-DCNN',
                #     '2017_mississipi_CSCRO_630_' + str(ii) + '.npz'
                # )
                # np.savez(res_file, test_res.eval())
                test_acc.append(acc)
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
            print('Testing time: {}'.format(time.time()-start_time))
            input('press Enter')
    # sys.stdout = orig_stdout
    # f.close()
            # save model
            if final_acc > 0.85:
                print('Saving models ...')
                saver = tf.train.Saver()
                model_dir = os.path.join(
                        home_dir,
                        'data_pool/U-TMP/excersize/ML_model/MC-DCNN',
                        'model_dir1015_'+str(final_acc)
                    )
                kk += 1
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                os.makedirs(model_dir)
                saver.save(sess, os.path.join(model_dir, "MultiChannel-DCNN"))
