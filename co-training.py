import os
import numpy as np
import random
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.decomposition import PCA


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


def co_training(model, pca, input_x, input_y, tau, num):
    if num == 1:
        train_x = input_x[:, :549]
    else:
        train_x = input_x[:, 549:]
    x = pca.transform(train_x)
    pred_y_mat = model.predict_proba(x)  # (n_samples, n_classes)
    pred_y = np.argmax(pred_y_mat, axis=1)
    # print(len(pred_y), len(input_y))
    acc1 = len(np.where(np.int64(pred_y) == input_y)[0])/float(len(input_y))
    print('Unlabeled data Train accuracy: {}'.format(acc1))
    add_x = []
    add_y = []
    good_idx = []
    flag = False
    for i in range(len(train_x)):
        ys = pred_y_mat[i]
        if max(ys) > tau:
            flag = True
            if num == 1:
                add_x.append(input_x[i, 549:])
            else:
                add_x.append(input_x[i, :549])
            add_y.append(np.argmax(ys))
            good_idx.append(i)
    out_x = np.delete(input_x, good_idx, 0)
    out_y = np.delete(input_y, good_idx, 0)
    return add_x, add_y, out_x, out_y, flag


if __name__ == "__main__":
    home_dir = os.path.expanduser('~')
    file = os.path.join(
        home_dir,
        'data_pool/waterfall_data/pretrain_result/yunjie-10/corn',
        '0401_0930_17_1_CoSoOt_L_REG_TEST_17.npz'
    )
    l_len = 500
    u_len = 10000
    t_len = 10000
    init_labeled_x, init_labeled_y = load_data(file, l_len)
    init_ulabeled_x, init_ulabeled_y = load_data(file, u_len)
    test_x, test_y = load_data(file, t_len)
    u_x1 = init_ulabeled_x
    u_x2 = u_x1
    u_y1 = init_ulabeled_y
    u_y2 = u_y1
    l_x1 = init_labeled_x[:, :549]
    l_x2 = init_labeled_x[:, 549:]
    l_y1 = init_labeled_y
    l_y2 = l_y1
    TAU = 0.9
    ii = 1
    P1 = True
    P2 = True
    while P1 or P2:
        model_clas1, model_pca1 = sub_model_train(l_x1, l_y1)
        model_clas2, model_pca2 = sub_model_train(l_x2, l_y2)
        new_labeled_x2, new_labeled_y2, nu_x1, nu_y1, P1 = co_training(
            model_clas1, model_pca1, u_x1, u_y1, TAU, 1
        )
        new_labeled_x1, new_labeled_y1, nu_x2, nu_y2, P2 = co_training(
            model_clas2, model_pca2, u_x2, u_y2, TAU, 2
        )
        if P2:
            l_x1 = np.concatenate((l_x1, np.array(new_labeled_x1)))
            l_y1 = np.concatenate((l_y1, np.array(new_labeled_y1)))
        if P1:
            l_x2 = np.concatenate((l_x2, np.array(new_labeled_x2)))
            l_y2 = np.concatenate((l_y2, np.array(new_labeled_y2)))
        if P1 or P2:
            print('train iteration: ', ii)
            ii += 1
            print('new labeled data set 1 : ', len(l_x1))
            print('new unlabeled data set 1 : ', len(nu_x1))
            print('new labeled data set 2 : ', len(l_x2))
            print('new unlabeled data set 2 : ', len(nu_x2))
        u_x1 = nu_x1
        u_y1 = nu_y1
        u_x2 = nu_x2
        u_y2 = nu_y2
    _y1 = model_clas1.predict(model_pca1.transform(test_x[:, :549]))
    _y2 = model_clas2.predict(model_pca2.transform(test_x[:, 549:]))
    print('self-training model acc: ', len(np.where(_y1 == test_y)[0])/float(t_len),
          len(np.where(_y2 == test_y)[0])/float(t_len))
    _model_clas1, _model_pca1 = sub_model_train(init_ulabeled_x[:, :549], init_ulabeled_y)
    _model_clas2, _model_pca2 = sub_model_train(init_ulabeled_x[:, 549:], init_ulabeled_y)
    _pca_x1 = _model_pca1.transform(test_x[:, :549])
    _pred_y1 = _model_clas1.predict(_pca_x1)
    _pca_x2 = _model_pca2.transform(test_x[:, 549:])
    _pred_y2 = _model_clas2.predict(_pca_x2)
    print('supervised model acc: ', len(np.where(_pred_y1 == test_y)[0])/float(t_len),
          len(np.where(_pred_y2 == test_y)[0]) / float(t_len))

