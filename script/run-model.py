import argparse
import pprint
from common import logger
from CROP_CLASSIFICATION.model.LSTM.lstm_main import main_stuff
from CROP_CLASSIFICATION.model.DecisionTree import DecisionTree
from CROP_CLASSIFICATION.model.SvmModel import SVMClassifier
from CROP_CLASSIFICATION.model.MLPClassifier import MLPNetWork


def do_stuff(
    model_type,
    tar_dir,
    data_file,
    batch_name,
    *args,
):
    logger.info("MODEL TRAINING {}".format(model_type))
    if model_type == "MLP":
        mlp = MLPNetWork(
            data_file,
            tar_dir,
            batch_name,
            init_learning_rate,
            max_iter,
            hidden_layer_sizes,
            test_size
        )
        mlp.model()
    elif model_type == "SVM":
        svm = SVMClassifier(
            data_file,
            tar_dir,
            batch_name,
            gamma,
            C,
            kernel,
        )
        svm.model()
    elif model_type == "DT":
        dt = DecisionTree(
            data_file,
            tar_dir,
            batch_name,
            max_depth,
            criterion,
        )
        dt.model()
    elif model_type == "LSTM":
        process_dict = {}
        process_dict["work_path"] = tar_dir
        process_dict["model_dir"] = batch_name
        process_dict["traindata_path_npz"] = data_file
        process_dict["chunk"] = chunk
        assert start_date[:4] == end_date[:4]
        year = start_date[:4]
        st_day = start_date[4:]
        ed_day = end_date[4:]
        process_dict["year_date"] = [year, st_day, ed_day]
        process_dict["training_steps"] = max_iter
        process_dict["init_learning_rate"] = init_learning_rate
        process_dict["num_classes"] = num_classes
        process_dict["dropout"] = dropout
        process_dict["num_hidden"] = num_hidden
        process_dict["num_layers"] = num_layers
        process_dict["decay"] = decay
        pprint.pprint(process_dict)
        main_stuff(process_dict, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Model training. Use -h for more information.")
    )

    sub_parser = parser.add_subparsers(
        description="Command for local training.", dest="use_mode")
    try_local_task1 = sub_parser.add_parser(
        "use-rnn", aliases=["ur"], help="Command for Long-Short term Memory model train.")

    try_local_task1.add_argument(
        metavar="target-dir",
        dest="tar_dir",
        help="The root path for model saving and also the direct path for saving processed data. e.g. ~/data2/citrus/demo/sample_result/TQLS/test/",
    )

    try_local_task1.add_argument(
        metavar="data-file",
        dest="data_file",
        help="Input file path. e.g. ~/data2/citrus/demo/sample_result/TQLS/TD_S3_L3a_20190319T114820_extract.npz",
    )

    try_local_task1.add_argument(
        metavar="batch-name",
        dest="batch_name",
        help="Combining with target-dir, the result will be the final path for model saving.",
    )

    try_local_task1.add_argument(
        "-model-type",
        type=str,
        default="LSTM",
        help="Model type, default LSTM.",
    )
    try_local_task1.add_argument(
        "-start-date",
        # dest="start_date",
        default='20180401',
        help="Start date, default '20180401'."
    )
    try_local_task1.add_argument(
        "-end-date",
        # dest="end_date",
        default='20180930',
        help="End date, default '20180930'."
    )

    try_local_task1.add_argument(
        metavar="lstm-state",
        dest="lstm_state",
        type=int,
        help="Data processing state, e.g. data process: 0, model train: 1, data process -> model train: 2.",
    )

    try_local_task1.add_argument(
        "-chunk-size",
        type=int,
        default=4000,
        help="The size of sliced date, default 4000 samples/.npz."
    )
    try_local_task1.add_argument(
        "-learning-rate-init",
        type=float,
        default=0.0001,
        help="The initial learning rate used, default 0.0001."
    )
    try_local_task1.add_argument(
        "-num-classes",
        type=int,
        default=2,
        help="Number of sample types. default 2."
    )
    try_local_task1.add_argument(
        "-max-iter",
        type=int,
        default=1000,
        help="Maximum number of iterations, default 1000."
    )
    try_local_task1.add_argument(
        "-dropout",
        type=float,
        default=0.4,
        help="The probability that each element is droped, default 0.4."
    )
    try_local_task1.add_argument(
        "-num-hidden-units",
        type=int,
        default=180,
        help="The number of net units per layer, default 180."
    )
    try_local_task1.add_argument(
        "-num-layers",
        type=int,
        default=1,
        help="The number of network layers, default 1."
    )
    try_local_task1.add_argument(
        "-learning-decay",
        type=float,
        default=0.96,
        help="The exponential decay for the learning rate, default 0.96."
    )

    try_local_task2 = sub_parser.add_parser(
        "use-mlp", aliases=["um"], help="Command for MLP model train.")
    try_local_task2.add_argument(
        metavar="target-dir",
        dest="tar_dir",
        help="The root path for model saving and also the direct path for saving processed data. e.g. ~/data2/citrus/demo/sample_result/TQLS/test/",
    )
    try_local_task2.add_argument(
        metavar="data-file",
        dest="data_file",
        help="Input file path. e.g. ~/data2/citrus/demo/sample_result/TQLS/test/TD_S3_L3a_20190319T114820_TRAIN.npz",
    )
    try_local_task2.add_argument(
        metavar="batch-name",
        dest="batch_name",
        help="Combining with target-dir, the result will be the final path for model saving.",
    )
    try_local_task2.add_argument(
        "-model-type",
        type=str,
        default="MLP",
        help="Model type, default MLP.",
    )
    try_local_task2.add_argument(
        "-hidden-layer-sizes",
        type=str,
        default="(100,)",
        help="The ith element represents the number of neurons in the ith hidden layer, please add \ before each parenthese, default (100,)."
    )
    try_local_task2.add_argument(
        "-learning-rate-init",
        type=float,
        default=0.001,
        help="The initial learning rate used, default 0.001. "
    )
    try_local_task2.add_argument(
        "-max_iter",
        type=int,
        default=200,
        help="Maximum number of iterations, default 200. "
    )
    try_local_task2.add_argument(
        "-test-size",
        type=float,
        default=0.3,
        help="The ratio of test data among all samples, default 0.3."
    )

    try_local_task3 = sub_parser.add_parser(
        "use-svm", aliases=["us"], help="Command for SVM model train.")
    try_local_task3.add_argument(
        metavar="target-dir",
        dest="tar_dir",
        help="The root path for model saving and also the direct path for saving processed data. e.g. ~/data2/citrus/demo/sample_result/TQLS/test/",
    )

    try_local_task3.add_argument(
        metavar="data-file",
        dest="data_file",
        help="Input file path. e.g. ~/data2/citrus/demo/sample_result/TQLS/test/TD_S3_L3a_20190319T114820_TRAIN.npz",
    )

    try_local_task3.add_argument(
        metavar="batch-name",
        dest="batch_name",
        help="Combining with target-dir, the result will be the final path for model saving.",
    )

    try_local_task3.add_argument(
        "-model-type",
        type=str,
        default="SVM",
        help="Model type, default SVM",
    )

    try_local_task3.add_argument(
        "-gamma",
        type=float,
        default=20,
        help="Kernel coefficient, default 20."
    )

    try_local_task3.add_argument(
        "-C",
        type=float,
        default=0.8,
        help="Penalty parameter C of the error term, default 0.8. "
    )

    try_local_task3.add_argument(
        "-kernel",
        type=str,
        default="rbf",
        help="Specifies the kernel type to be used in the algorithm, default 'rbf'."
    )

    try_local_task4 = sub_parser.add_parser(
        "use-dt", aliases=["ud"], help="Command for Decision Tree model train.")
    try_local_task4.add_argument(
        metavar="target-dir",
        dest="tar_dir",
        help="The root path for model saving and also the direct path for saving processed data. e.g. ~/data2/citrus/demo/sample_result/TQLS/test/",
    )
    try_local_task4.add_argument(
        metavar="data-file",
        dest="data_file",
        help="Input file path. e.g. ~/data2/citrus/demo/sample_result/TQLS/test/TD_S3_L3a_20190319T114820_TRAIN.npz",
    )
    try_local_task4.add_argument(
        metavar="batch-name",
        dest="batch_name",
        help="Combining with target-dir, the result will be the final path for model saving.",
    )
    try_local_task4.add_argument(
        "-model-type",
        type=str,
        default="DT",
        help="Model type, default DT",
    )
    try_local_task4.add_argument(
        "-max-depth",
        type=int,
        default=10,
        help="The maximum depth of the tree, default 10. "
    )
    try_local_task4.add_argument(
        "-criterion",
        type=str,
        default="gini",
        help="The function to measure the quality of a split, default 'gini'."
    )

    args = parser.parse_args()
    batch_name = args.batch_name
    model_type = args.model_type
    data_file = args.data_file
    tar_dir = args.tar_dir

    print(args)
    if args.use_mode in ["use-rnn", "ur"]:
        # LSTM model parameters
        state = args.lstm_state
        start_date = args.start_date
        end_date = args.end_date
        chunk = args.chunk_size
        init_learning_rate = args.learning_rate_init
        num_classes = args.num_classes
        max_iter = args.max_iter
        dropout = args.dropout
        num_hidden = args.num_hidden_units
        num_layers = args.num_layers
        decay = args.learning_decay
        do_stuff(
            model_type,
            tar_dir,
            data_file,
            batch_name,
            chunk,
            start_date,
            end_date,
            state,
            init_learning_rate,
            num_classes,
            max_iter,
            dropout,
            num_hidden,
            num_layers,
            decay,
        )
    elif args.use_mode in ["use-mlp", "um"]:
        # MLP model parameters
        hidden_layer_sizes = eval(args.hidden_layer_sizes)
        assert isinstance(hidden_layer_sizes, tuple)
        test_size = args.test_size
        max_iter = args.max_iter
        init_learning_rate = args.learning_rate_init
        do_stuff(
            model_type,
            tar_dir,
            data_file,
            batch_name,
            init_learning_rate,
            max_iter,
            hidden_layer_sizes,
            test_size
        )
    elif args.use_mode in ["use-svm", "us"]:
        # SVM model parameters
        gamma = args.gamma
        C = args.C
        kernel = args.kernel
        do_stuff(
            model_type,
            tar_dir,
            data_file,
            batch_name,
            gamma,
            C,
            kernel,
        )
    elif args.use_mode in ["use-dt", "ud"]:
        # DT model parameters
        max_depth = args.max_depth
        criterion = args.criterion
        do_stuff(
            model_type,
            tar_dir,
            data_file,
            batch_name,
            max_depth,
            criterion,
        )
