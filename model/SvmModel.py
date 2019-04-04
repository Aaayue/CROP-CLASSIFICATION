import numpy as np
import os
import time
import logging
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


class SVMClassifier:
    logging.basicConfig(level=logging.DEBUG)
    my_logger = logging.getLogger(__name__)

    def __init__(
        self,
        data_file: str,
        model_folder: str,
        model_dir: str,
        gamma: float = 20,
        C: float = 0.8,
        kernel: str = "rbf",
        *,
        label_str="",
        decision_function_shape: str = "ovr",
        train_ratio: float = 0.6,
        vc_ratio: float = 0.2,
        crossValidation_num: int = 3,
    ):

        # initialize parameters
        assert 0 < train_ratio < 1, "train_ratio must be in (0, 1)"
        assert 0 < vc_ratio < 1, "validation_ratio must be in (0, 1)"
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.decision_function_shape = decision_function_shape

        self.data_file = data_file
        self.model_folder = model_folder
        self.crossValidation_num = crossValidation_num
        self.train_ratio = train_ratio
        self.vc_ratio = vc_ratio
        self.label_str = label_str
        self.model_dir = model_dir

    def model(self) -> str:
        """
        SVM classifier
        warning: this model will work very slow
        if there are a lot of features in samples
        """

        self.my_logger.info("SVM model training...")
        data = np.load(self.data_file)
        train_feature = data['features']
        train_label = data['labels']
        print("feature shape: ", train_feature.shape)
        print("label shape: ", train_label.shape)

        # save model
        path = os.path.join(self.model_folder, self.model_dir +
                            '_' + time.strftime("%Y%m%dT%H%M%S"))
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = (
            "SVM_" + self.label_str +
            time.strftime("%Y%m%d%H%M%S", time.localtime()) + ".m"
        )

        model_path = os.path.join(path, model_name)

        # build svm classifier
        model = svm.SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            decision_function_shape=self.decision_function_shape,
            verbose=True,
        )
        for loop_index in range(self.crossValidation_num):
            print("No.", loop_index + 1)
            (train_sub_feature,
             test_sub_feature,
             train_sub_label,
             test_sub_label) = train_test_split(
                train_feature,
                train_label,
                test_size=self.vc_ratio,
                train_size=self.train_ratio
            )
            model = model.fit(train_sub_feature, train_sub_label)

            print(
                "Training accuracy: %f"
                % (model.score(train_sub_feature, train_sub_label))
            )

            print(
                "Testing accuracy: %f" % (
                    model.score(test_sub_feature, test_sub_label))
            )
            print(
                "Confusion matrix:\n",
                confusion_matrix(
                    test_sub_label, model.predict(test_sub_feature)),
            )
            f1 = f1_score(test_sub_label, model.predict(test_sub_feature))
            print(
                "F1 score: %f" % (f1)
            )

        print("SVM fitting...")
        model.fit(train_feature, train_label)

        try:
            joblib.dump(model, model_path)
            self.my_logger.info(
                "SVM model saved! Result path: %s", model_path)

        except Exception as e:
            self.my_logger.error(
                "{}, Save SVM model failed!".format(e))
            return None

        return model_path


if __name__ == '__main__':
    file = '/home/zy/data2/citrus/demo/sample_result/125040_noDEM_20190307T204448/TD_S3_L3a_20190307T204448_TRAIN.npz'
    SVM = SVMClassifier(file, "/home/zy/data_pool/U-TMP/TMP",
                        train_ratio=0.5, C=800, gamma=0.1)
    SVM.model()
