import os
import time
import logging
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class DecisionTree():
    logging.basicConfig(level=logging.DEBUG)
    my_logger = logging.getLogger(__name__)

    def __init__(
        self,
        data_file: str,
        model_folder: str,
        model_dir: str,
        max_depth: int = 10,
        criterion_type: str = "gini",
        *,
        crossValidation_num: int = 3,
        train_ratio: float = 0.6,
        vc_ratio: float = 0.2,
        label_str: str = "",
    ):
        # check if all parameters are valid
        assert 0 < train_ratio < 1, "train_ratio must be in (0, 1)"
        assert 0 < vc_ratio < 1, "validation_ratio must be in (0, 1)"
        assert 5 <= max_depth <= 100, "max_depth must be in [5, 100]"
        assert criterion_type in [
            "gini",
            "entropy",
        ], "criterion_type must be 'gini' or 'entropy'"
        assert os.path.exists(model_folder), "model_folder must be existed"
        assert 0 <= crossValidation_num <= 10, "crossValidation_num must be in [0, 10]"
        self.data_file = data_file
        self.model_folder = model_folder
        self.crossValidation_num = crossValidation_num
        self.train_ratio = train_ratio
        self.vc_ratio = vc_ratio
        self.max_depth = max_depth
        self.criterion_type = criterion_type
        self.label_str = label_str
        self.model_dir = model_dir

    def model(self) -> str:
        """
        Function:
            Training decision tree model
        Input:
            training_data: npz file, features=np.ndarray, labels=np.ndarray
            model_folder: string, folder path to save the model file
            crossValidation_num: int, optional (default = 3), number of cross validation
            train_ratio: float, optional (default = 0.2), ratio of record number
                        in cross validation
            max_depth: int, optional (default = 10), max depth of decision tree
            criterion_type: string, optional (default = "entropy")
                            “gini” for the Gini impurity and “entropy” for the
                            information gain
            label_str: string, optional (default = ""), result model name label
        Output:
            model_path: string, full path of decision tree model file
                        if is None, training model failed
        """

        self.my_logger.info("Decision tree model training...")

        # split train feature and label
        data = np.load(self.data_file)
        train_feature = data['features']
        train_label = data['labels']

        print("feature shape: ", train_feature.shape)
        print("label shape: ", train_label.shape)

        # cross validation
        self.my_logger.info("Cross validation...")
        # validation here just to make sure the model is stable,
        # not for seleting parameters
        for loop_index in range(self.crossValidation_num):
            print("No.", loop_index + 1)
            (train_sub_feature,
             test_sub_feature,
             train_sub_label,
             test_sub_label) = train_test_split(
                 train_feature, train_label,
                 test_size=self.vc_ratio,
                 train_size=self.train_ratio
            )
            DT_model = tree.DecisionTreeClassifier(
                max_depth=self.max_depth, criterion=self.criterion_type
            )
            DT_model = DT_model.fit(train_sub_feature, train_sub_label)

            # print("Features weight:", (DT_model.feature_importances_))

            print(
                "Training accuracy: %f"
                % (DT_model.score(train_sub_feature, train_sub_label))
            )

            print(
                "Testing accuracy: %f" % (DT_model.score(
                    test_sub_feature, test_sub_label))
            )
            print(
                "Confusion matrix:\n",
                confusion_matrix(
                    test_sub_label, DT_model.predict(test_sub_feature)),
            )

        # get final model
        self.my_logger.info("Final model training...")
        DT_model = tree.DecisionTreeClassifier(
            max_depth=self.max_depth, criterion=self.criterion_type
        )
        DT_model = DT_model.fit(train_feature, train_label)
        print("Training accuracy:%f" %
              (DT_model.score(train_feature, train_label)))

        # save model
        path = os.path.join(self.model_folder, self.model_dir +
                            '_' + time.strftime("%Y%m%dT%H%M%S"))
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = (
            "DecisionTree_"
            + self.label_str
            + time.strftime("%Y%m%d%H%M%S", time.localtime())
            + ".m"
        )
        model_path = os.path.join(path, model_name)

        try:
            joblib.dump(DT_model, model_path)
            self.my_logger.info(
                "Decision tree model saved! Result path: %s", model_path)

        except Exception as e:
            self.my_logger.error(
                "{}, Save decision tree model failed!".format(e))
            return None

        return model_path


if __name__ == "__main__":
    file = '/home/zy/data2/citrus/demo/sample_result/125040_noDEM_20190307T204448/TD_S3_L3a_20190307T204448_TRAIN.npz'
    DT = DecisionTree(
        file, "/home/zy/data_pool/U-TMP/TMP", criterion_type='gini')
    DT.model()
