import os
import time
import tqdm
import random
import theano
import tkinter
import theano.tensor as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from batchNormalization import BatchNormLayer
import lasagne
from lasagne.layers import InputLayer, InverseLayer, DropoutLayer, DenseLayer, Conv1DLayer, MaxPool1DLayer, \
    Upscale1DLayer, ReshapeLayer
import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', module='.*lasagne.init.*')
warnings.filterwarnings('ignore', module='.*nolearn.lasagne.*')
warnings.filterwarnings('ignore')


class Param(object):

    def __init__(self):
        self.N_USERS = 10
        self.N_CHANNELS = 6
        self.TEST_TRAIN_SPLIT = 0.80  # fraction of data for train/test
        # low pass filter requirements.
        self.ORDER = 1
        self.FS = 1.0  # approx sample rate, Hz
        self.CUTOFF = 0.00046  # desired cutoff frequency,Hz(~30 minute periods)
        self.SUBSAMPLE = 200
        # spectrogram requirement:
        self.FREQ_WINDOW = 91  # size of the sliding window
        # self.STEP = self.FREQ_WINDOW - 400


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def load_data(npz_file):
    a = np.load(npz_file)
    feat = a['features']
    feat = np.array(feat)
    lab = a['labels']
    smp_feat = feat
    smp_lab = lab
    nums = int(0.7 * len(lab))
    train_idx = random.sample(range(len(lab)), nums)
    test_idx = np.delete(range(len(lab)), train_idx)
    train_feat = smp_feat[train_idx, :]
    train_label = smp_lab[train_idx].reshape(-1)
    test_feat = smp_feat[test_idx, :]
    test_label = smp_lab[test_idx].reshape(-1)
    train_feat = train_feat.reshape(-1, 6, 91)
    test_feat = test_feat.reshape(-1, 6, 91)
    print(train_feat.shape, test_feat.shape)
    return train_feat, train_label, test_feat, test_label


def batch_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    layer = BatchNormLayer(layer, **kwargs)
    if nonlinearity is not None:
        from lasagne.layers import NonlinearityLayer
        layer = NonlinearityLayer(layer, nonlinearity)
    return layer


batch_size = None
parameters = Param()


def build_MC_DCNN(input_var=[None] * parameters.N_CHANNELS):
    conv_num_filters1 = 4
    conv_num_filters2 = 4
    filter_size1 = 5
    filter_size2 = 5
    pool_size = 4
    pad_in = 'valid'
    pad_out = 'full'
    dense_units = 32
    ########################
    # Here we build a dictionnary to construct independent layers for each channel:
    network = {}
    for i in range(parameters.N_CHANNELS):
        network[i] = InputLayer(shape=(batch_size, 1, parameters.FREQ_WINDOW), input_var=input_var[i],
                                name="input_layer_1")

        network[i] = batch_norm(Conv1DLayer(
            network[i], num_filters=conv_num_filters1, filter_size=filter_size1,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform(), pad=pad_in, name="conv1_1"))

        network[i] = MaxPool1DLayer(network[i], pool_size=pool_size, name="pool1_1")

        network[i] = batch_norm(Conv1DLayer(
            network[i], num_filters=conv_num_filters2, filter_size=filter_size2,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform(), pad=pad_in, name="conv2_1"))

        network[i] = MaxPool1DLayer(network[i], pool_size=pool_size, name="pool2_1")

        network[i] = ReshapeLayer(network[i], shape=([0], -1), name="reshape_1")

    #######################
    # Now we concatenate the output from each channel layer, and build a MLP:

    network2 = lasagne.layers.ConcatLayer(network.values(), axis=1, name="concat")

    network2 = batch_norm(lasagne.layers.DenseLayer(network2, num_units=dense_units,
                                                    W=lasagne.init.GlorotUniform(), name="dense1"))

    network2 = batch_norm(lasagne.layers.DenseLayer(network2, num_units=parameters.N_USERS - 1,
                                                    W=lasagne.init.GlorotUniform(),
                                                    nonlinearity=lasagne.nonlinearities.softmax, name="output"))

    return network2


start = time.time()
# The target values (correct classes) are stored a one hot matrix, of size Training_samples x classes
target_values = T.lmatrix('target_output')
# target_values = T.imatrix('target_output')
# we then build a dictionary of input variables (one per channel)
inps = {}
for i in range(parameters.N_CHANNELS):
    inps[i] = T.tensor3()

# Let's build the MC_DCNN, and check the architecture by printing out the layer sizes and names:
# network = build_MC_DCNN(inps.values())
network = build_MC_DCNN(list(inps.values()))

laylist = lasagne.layers.get_all_layers(network)

for l in laylist:
    print(l.name, lasagne.layers.get_output_shape(l))

num_params = lasagne.layers.count_params(network)

print("number of parameters is {}".format(num_params))

# lasagne.layers.get_output produces an expression for the output of the net:
network_output = lasagne.layers.get_output(network)
# print(type(network_output), network_output)

# Our cost will be categorical cross-entropie
cost = lasagne.objectives.categorical_crossentropy(network_output, target_values)
cost = cost.mean()

# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(network, trainable=True)
# Compute adam updates for training
updates = lasagne.updates.adam(cost, all_params)
# Theano functions for training and computing cost.
print(list(inps.values()))
# print(target_values)
train = theano.function(list(inps.values())+[target_values], cost, updates=updates)
# train = theano.function(inps.values() + [target_values], cost, updates=updates)
compute_cost = theano.function(list(inps.values()) + [target_values], cost)

# Theano functions for forward pass:
predicted_values = lasagne.layers.get_output(network, deterministic=True)
predict = theano.function(list(inps.values()), [predicted_values])

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
num_epochs = 200
home_dir = os.path.expanduser('~')
data_file = os.path.join(
    home_dir,
    'data_pool/waterfall_data/pretrain_result/yunjie/mississipi',
    '0401_0630_17_1_CoSoOtCoRi_L_REG_TEST_17.npz'
)
X_train, y_train, X_test, y_test = load_data(data_file)
y_train = label_binarize(y_train, classes=range(parameters.N_USERS-1)).astype('int64')
y_test = label_binarize(y_test, classes=range(parameters.N_USERS-1)).astype('int64')
error_train = []
error_val = []
for epoch in tqdm(range(num_epochs)):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    batch_siz = 200
    for batch in iterate_minibatches(X_train, y_train, batch_siz, shuffle=True):
        inputs, targets = batch
        # Here we splits the input into individual channels:
        # inputs2 = [inputs[:, i, :].reshape(-1, parameters.FREQ_WINDOW) for i in range(parameters.N_CHANNELS)]
        inputs2 = [inputs[:, i, :].reshape(-1, 1, parameters.FREQ_WINDOW) for i in range(parameters.N_CHANNELS)]
        argList = inputs2 + [targets]

        train_err += train(*argList)
        print(train_err)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_siz, shuffle=False):
        inputs, targets = batch
        inputs2 = [inputs[:, i, :].reshape(-1, 1, parameters.FREQ_WINDOW) for i in range(parameters.N_CHANNELS)]
        argList = inputs2 + [targets]

        val_err += compute_cost(*argList)
        val_batches += 1

    error_train += [train_err / train_batches]
    error_val += [val_err / val_batches]

    # Each epoch, we do some predictions on the test data and compute the F1 score:
    inputs_pred = [X_test[:, i, :].reshape(-1, 1, parameters.FREQ_WINDOW) for i in range(parameters.N_CHANNELS)]

    y_pred = [y.argmax() for y in predict(*inputs_pred)[0]]
    y_true = [y.argmax() for y in y_test]
    # print("F1 score {}".format(f1_score(y_true, y_pred)))
    conf_mat_test = confusion_matrix(y_true, y_pred)
    print('testing confusion matrix:')
    print(conf_mat_test)
    acc = (conf_mat_test[0, 0]+conf_mat_test[1, 1]+conf_mat_test[2, 2]+conf_mat_test[3, 3]+conf_mat_test[4, 4])/64303
    print('testing accuracy: ', acc)
    print('processing time: ', time.time()-start)
    # Let's plot the training/testing error
    # plt.plot(error_train)
    # plt.plot(error_val)
    # plt.title("F1 score {}".format(f1_score(y_true, y_pred)))
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    # plt.close()
