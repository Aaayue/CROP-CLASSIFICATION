import os
import time
import glob
import json
import socket
import numpy as np
import pandas as pd
from CROP_CLASSIFICATION.script.common import logger
from os.path import join
from CROP_CLASSIFICATION.model.LSTM.LSTModel import LSTModel
from CROP_CLASSIFICATION.model.LSTM.TrainDataFlat import batch_run
from CROP_CLASSIFICATION.model.LSTM.TrainDataSG import TrainDataSG
logger.handlers.pop()


def save_json(file, data):
    with open(file, 'w') as f:
        json.dump(data, f)


def data_process(process_dict):
    """
    time series SG filtering and data flatten
    """
    sg_file = process_dict["traindata_path_npz"]
    if '.npz' not in sg_file:
        raise Exception('Input must be NPZ format')

    SG = TrainDataSG(
        file=sg_file,
        save_path=process_dict['work_path'],
        quantity=process_dict["chunk"],
        year=process_dict['year_date'][0],
        start_day=process_dict['year_date'][1],
        end_day=process_dict['year_date'][2]
    )
    SG.batch_run()
    sg_list = glob.glob(join(process_dict["work_path"], '*.npz'))
    sg_list = [s for s in sg_list if '_' +
               str(process_dict["chunk"]) + '_' in s]

    if len(sg_list) == 0:
        raise Exception("Invalid time series data in samples")
    process_dict["traindata_path_npz"] = batch_run(sg_list)
    # save process dict
    result_file = join(
        process_dict["work_path"],
        "traindata_" + time.strftime("%Y%m%dT%H%M%S") + ".json",
    )
    save_json(result_file, process_dict)
    logger.info("Extract data finish!")
    return process_dict


def model_train(process_dict):
    """
    using training data to run model
    """
    # get training data
    train_file = process_dict["traindata_path_npz"]
    traindata = np.load(train_file)
    train_lab = traindata['labels'].tolist()
    feature_length = traindata["features"].shape[1]
    logger.info("total label: %d" % sum(train_lab))

    model_dir = join(
        process_dict["work_path"],
        process_dict['model_dir'] + '_' + time.strftime("%Y%m%dT%H%M%S")
    )
    timestep = pd.date_range(
        process_dict['year_date'][0] + process_dict['year_date'][1],
        process_dict['year_date'][0] + process_dict['year_date'][2],
        freq="5D"
    )

    logger.info('feature length {} and dates {}'.format(
        feature_length, len(timestep)))
    lstm = LSTModel(
        model_dir=model_dir,
        file=train_file,
        num_input=int(feature_length / len(timestep)),
        timesteps=len(timestep),
        training_steps=process_dict["training_steps"],
        init_learning_rate=process_dict["init_learning_rate"],
        num_classes=process_dict["num_classes"],
        dropout=process_dict["dropout"],
        num_hidden=process_dict["num_hidden"],
        num_layers=process_dict["num_layers"],
        decay=process_dict["decay"],

    )
    process_dict["model_path"] = lstm.sess_run()

    # save process dict
    result_file = join(
        process_dict["work_path"],
        "model_" + time.strftime("%Y%m%dT%H%M%S") + ".json"
    )

    save_json(result_file, process_dict)
    logger.info("Training model finish!")
    return process_dict


def main_stuff(process_dict, process_state):
    """
    process_state:
    0: data_process
    1: model_train
    2: data_process -> model_train
    """
    process_ins = {
        0: 'DATA PROCCESS',
        1: 'MODEL TRAIN',
        2: 'DATA PROCESS -> MODEL TRAIN',
    }
    if process_state in [0, 2]:
        res_dict = data_process(process_dict)
        if process_state == 2:
            res_dict = model_train(res_dict)
            logger.info('ALL DONE! {}'.format(process_ins[process_state]))
        else:
            logger.info('ALL DONE! {}'.format(process_ins[process_state]))
    if process_state == 1:
        res_dict = model_train(process_dict)
        logger.info('ALL DONE! {}'.format(process_ins[process_state]))


if __name__ == '__main__':
    home_dir = os.path.expanduser('~')
    process_dict = {
        "work_path": "",
        "model_path": "",
        "traindata_path_npz": "",
        "res_label": "hn_125_Demo_0227",
        # ===========================
        "model_dir": "lstm_bn",
        "chunk": 4000,
        "year_date": ['2018', '0101', '1231'],
        # ===========================
    }
    state = 2
    if state in [0, 2]:
        # make work path
        work_path = join(
            home_dir, "data2/citrus/demo/sample_result/TQLS/test/")

        process_dict['traindata_path_npz'] = join(
            home_dir,
            "data2/citrus/demo/sample_result/TQLS/sichuan/"
            + "arfar-megatron_48RVU_tqls_20190319T114813/"
            + "TD_S3_L3a_20190319T114820_extract.npz")

        print('PROCESS DATA ', process_dict['traindata_path_npz'])
        process_dict["work_path"] = join(work_path, "{}_{}/".format(
            socket.gethostname(), time.strftime("%Y%m%dT%H%M%S")))

        if not os.path.exists(process_dict["work_path"]):
            os.makedirs(process_dict["work_path"])

        main_stuff(process_dict, state)

    if state == 1:
        json_file = join(
            home_dir,
            'data2/citrus/demo/sample_result/'
            + 'TQLS/sichuan/merge_sample/traindata_merge_sample.json')
        with open(json_file) as f:
            process_dict = json.load(f)
        main_stuff(process_dict, state)
