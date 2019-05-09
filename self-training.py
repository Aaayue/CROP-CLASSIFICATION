import os
import time
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# from sklearn.multiclass import OneVsRestClassifier


def load_data(npz_file, num):
    a = np.load(npz_file)
    feat = a['features']
    lab = a['labels']
    lab[lab == np.int64(6)] = np.int64(2)
    idx = random.sample(range(len(lab)), num)
    return feat[idx], lab[idx]


def sub_model_train(x1, y1):
    pca = PCA(
        n_components=1-1e-7,  # 最好设置为方差保留率
        copy=True,
        whiten=False,
        svd_solver='auto',
        # tol=0.00001,
        iterated_power='auto'
    )
    print('=========================================================================')
    print('start processing PCA...')
    print(x1.shape)
    pca.fit(x1)
    pca_x1 = pca.transform(x1)
    print('finishing PCA...')
    print('PCA model:', pca)
    print('PCA results:', pca_x1.shape)
    clf = SVC(
        C=800,
        gamma=0.1,
        kernel='rbf',
        decision_function_shape='ovr',
        max_iter=-1,
        cache_size=1000,
        tol=0.001,
        probability=True,
        # verbose=True
    )
    model = clf.fit(pca_x1, y1)
    return model, pca


def self_training(model, pca, input_x, input_y, tau):
    x = pca.transform(input_x)
    pred_y_mat = model.predict_proba(x)  # (n_samples, n_classes)
    # print(pred_y_mat[:10])
    pred_y = np.argmax(pred_y_mat, axis=1)
    acc1 = len(np.where(np.int64(pred_y) == input_y)[0])/float(len(input_y))
    print('Unlabeled data Train accuracy: {}'.format(acc1))
    good_x = []
    good_y = []
    good_idx = []
    flag = False
    for i in range(len(input_x)):
        ys = pred_y_mat[i]
        if max(ys) > tau:
            flag = True
            good_x.append(input_x[i])
            good_y.append(np.argmax(ys))
            good_idx.append(i)
    out_x = np.delete(input_x, good_idx, 0)
    out_y = np.delete(input_y, good_idx, 0)
    return good_x, good_y, out_x, out_y, flag


if __name__ == "__main__":
    home_dir = os.path.expanduser('~')
    file = os.path.join(
        home_dir,
        'data_pool/waterfall_data/pretrain_result/yunjie-10/corn',
        '0401_0930_17_1_CoSoOt_L_REG_TEST_17.npz'
    )
    l_len = 500
    u_len = 20000
    t_len = 10000
    init_labeled_x, init_labeled_y = load_data(file, l_len)
    init_ulabeled_x, init_ulabeled_y = load_data(file, u_len)
    test_x, test_y = load_data(file, t_len)
    u_x = init_ulabeled_x
    u_y = init_ulabeled_y
    l_x = init_labeled_x
    l_y = init_labeled_y
    # svm, pca = sub_model_train(init_labeled_x, init_labeled_y)
    TAU = 0.95
    ii = 1
    P = True
    while P:
        model_svm, model_pca = sub_model_train(l_x, l_y)
        new_labeled_x, new_labeled_y, u_x, u_y, P = self_training(model_svm, model_pca, u_x, u_y, TAU)
        if P:
            l_x = np.concatenate((l_x, np.array(new_labeled_x)))
            l_y = np.concatenate((l_y, np.array(new_labeled_y)))
            print('train iteration: ', ii)
            ii += 1
            print('new labeled data set: ', len(l_x))
            print('new unlabeled data set: ', len(u_x))
    _y = model_svm.predict(model_pca.transform(test_x))
    print('self-training model acc: ', len(np.where(_y == test_y)[0])/float(t_len))
    _model_svm, _model_pca = sub_model_train(init_ulabeled_x, init_ulabeled_y)
    _pca_x = _model_pca.transform(test_x)
    _pred_y = _model_svm.predict(_pca_x)
    print('supervised model acc: ', len(np.where(_pred_y == test_y)[0])/float(t_len))





