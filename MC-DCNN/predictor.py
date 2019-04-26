
import numpy as np
from os.path import join
import tensorflow as tf
import os
import time

class Predicter:
    def __init__(self, model_dir, tf_session=None):
        if tf_session is None:
            conf = tf.ConfigProto()
            conf.gpu_options.allow_growth = True
            tf_session = tf.Session(config=conf)
        saver = tf.train.import_meta_graph(join(model_dir, "MultiChannel-DCNN.meta"))
        saver.restore(tf_session, tf.train.latest_checkpoint(model_dir))
        self.tf_session = tf_session

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.tf_session.close()

    def predict_the_fuck(self, features_predict):
        """
        Args:
        ----
            features_predict: 3-d, [pixelNum, bandNum, timeSeriesNum]
        """

        # features_predict = features_predict.transpose(0, 2, 1) / 10000

        prediction = self.tf_session.run(
            ["predict_results:0"], feed_dict={"x-input:0": features_predict}
        )

        prediction = prediction[0]
        crop_type_index = [np.argmax(result) for result in prediction]

        return crop_type_index


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


if __name__ == "__main__":
    home_dir = os.path.expanduser('~')
    model_dir = join(home_dir, 'data_pool/U-TMP/excersize/ML_model/MC-DCNN/model_dir_1')
    # print(model_dir)
    P = Predicter(model_dir)
    # data_file = join(
    #     home_dir,
    #     'data_pool/waterfall_data/pretrain_result/yunjie/mississipi',
    #     '0401_0630_17_1_CoSoOtCoRi_L_REG_TRAIN_141516.npz'
    # )
    data_file = join(
        home_dir,
        'data_pool/waterfall_data/pretrain_result/yunjie/mississipi',
        '0401_0630_17_1_CoSoOtCoRi_L_REG_TEST_17.npz'
    )
    data = np.load(data_file)
    x_data = data['features']
    # shape is (None, 546)
    x_data = x_data.reshape(-1, 6, 91)
    print(x_data.shape)
    y_data = data['labels']
    y_data[y_data == np.int64(6)] = np.int64(4)
    ERR = 0
    for batch in iterate_minibatches(x_data, y_data, 200, shuffle=False):
        start = time.time()
        part_x, part_y = batch
        part_x_chan = [part_x[:, k, :].reshape((-1, 1, 91)) for k in range(6)]
        pred_y = P.predict_the_fuck(part_x_chan)
        err = len(np.where(pred_y != part_y)[0])
        if err > 100:
            print('predict results: ', pred_y)
            print('true label: ', part_y)
            input('press Enter')
        # print(err)
        ERR += err
    print('total data: ', len(y_data))
    print('total error: ', ERR)
    print('ACCURACY: ', 1-(ERR/float(len(y_data))))
    print('test confusion matrix:')
    print(tf.confusion_matrix(pred_y, part_y, 5))
    print("batch precess time: ", time.time()-start)
