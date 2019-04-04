import os
import time
import socket
import pprint
import numpy as np
import pandas as pd
from common import logger
from baikal.model.RunPredictor import run_predictor
from baikal.model.LSTM import LSTModel
from baikal.general.common import (
    save_json,
    load_json,
    get_bands_into_a_list,
    combine_npys,
    get_bands_into_a_dict,
    set_vvdic_key_order,
)
from baikal.ground_truth.TrainDataExtractor import TrainDataExtractorV2
from baikal.ground_truth.TrainDataSG import TrainDataSG
from baikal.ground_truth.TrainDataFlat import batch_run
from os.path import join

# for model test
import glob
import json


def go_get_training(process_dict: dict) -> dict:
    """
    using label shape and tif to get training data
    """
    # get all need data for processing
    process_dict["img_pro_dict"] = get_bands_into_a_dict(
        process_dict, "*.tif"
    )

    # get a read order list
    read_list, time_list = set_vvdic_key_order(
        process_dict["img_pro_dict"]
    )
    process_dict["read_order_list"] = read_list
    process_dict["time_order_list"] = time_list

    pprint.pprint(process_dict)
    # run get training data
    tde = TrainDataExtractorV2(
        process_dict,
        sample_label="S3_L3a_" + time.strftime("%Y%m%dT%H%M%S"),
        label_keep_list=[1],
        is_binarize=True,
    )
    traindata_dict, trainlab, process_dict[
        "traindata_path_npz"
    ] = tde.go_get_mask_2npy()
    sg_file = process_dict["traindata_path_npz"]
    print(sg_file)
    SG = TrainDataSG(
        file=sg_file,
        quantity=process_dict["chunk"],
        year=process_dict['year_date'][0],
        start_day=process_dict['year_date'][1],
        end_day=process_dict['year_date'][2]
    )
    SG.batch_run()
    sg_list = glob.glob(join(process_dict["work_path"], '*.npz'))
    sg_list = [s for s in sg_list if '_' +
               str(process_dict["chunk"]) + '_' in s]
    pprint.pprint(sg_list)
    if len(sg_list) == 0:
        raise Exception("invalid time series data in this tile")
    process_dict["traindata_path_npz"] = batch_run(sg_list)
    # save process dict
    result_file = os.path.join(
        process_dict["work_path"],
        "traindata_" + time.strftime("%Y%m%dT%H%M%S") + ".json",
    )
    save_json(result_file, process_dict)
    print("Extract data finish!")
    return process_dict


def go_to_training(process_dict: dict) -> dict:
    """
    using training data to run model
    """
    # get training data
    train_file = process_dict["traindata_path_npz"]
    traindata = np.load(train_file)
    train_lab = traindata['labels'].tolist()
    feature_length = traindata["features"].shape[1]
    print("total label: %d" % sum(train_lab))

    # TODO: get model save file from process dict
    # model_dir = os.path.join(
    #     process_dict["work_path"], process_dict["model_dir"])
    model_dir = os.path.join(
        process_dict["work_path"], 'lstm_bn_' + time.strftime("%Y%m%dT%H%M%S"))
    timestep = pd.date_range(
        process_dict['year_date'][0] + process_dict['year_date'][1],
        process_dict['year_date'][0] + process_dict['year_date'][2],
        freq="5D")

    print(feature_length, len(timestep))
    LSTM = LSTModel(
        model_dir=model_dir,
        file=train_file,
        num_input=int(feature_length / len(timestep)),
        timesteps=len(timestep),
    )
    process_dict["model_path"] = LSTM.sess_run()

    # save process dict
    result_file = os.path.join(
        process_dict["work_path"],
        "model_" + time.strftime("%Y%m%dT%H%M%S") + ".json"
    )

    save_json(result_file, process_dict)
    print("Training model finish!")
    return process_dict


def go_to_predictor(process_dict: dict) -> dict:
    """
    using model ro run prodict
    """
    # run predictor
    status, process_dict["result_path"] = run_predictor(process_dict)
    if status:
        print("result file: ", process_dict["result_path"])
    else:
        print("Failed!")
    pprint.pprint(process_dict)

    # save process dict
    result_file = os.path.join(
        process_dict["work_path"], "product_" +
                                   time.strftime("%Y%m%dT%H%M%S") + ".json"
    )
    save_json(result_file, process_dict)
    print("Predict result finish!")
    return process_dict


def control_tool(json_file, process_dict, process_label=0) -> bool:
    """
    Function:
        0: get training data -> model training -> run prodict
        1: get training data -> model training
        2: get training data
        3: model training
            must be give the training process dict
        4: run prodict
            must be give the product
    """
    if process_label in [0, 1, 2]:
        process_dict_training = go_get_training(process_dict)
        if process_label in [0, 1]:
            process_dict_model = go_to_training(process_dict_training)
            if process_label == 0:
                process_dict_product = go_to_predictor(process_dict_model)
                print("product finished! %d" % process_label)
            else:
                print("product finished! %d" % process_label)
        else:
            print("product finished! %d" % process_label)
    elif process_label == 3:
        with open(json_file) as f:
            process_dict = json.load(f)
        process_dict_model = go_to_training(process_dict)
        print("model training finished! %d" % process_label)
    elif process_label == 4:
        process_dict_product = go_to_predictor(process_dict)
        print("product finished! %d" % process_label)
    else:
        print("please check the process label!")


if __name__ == "__main__":
    # for input
    # DEM source data must be the first source in process_dict
    print('start')
    home_dir = os.path.expanduser('~')
    process_dict = {
        "ori_ras_path": {},
        "img_pro_dict": {},
        "pro_ras_list": [{
            "Sentinel_1": join(home_dir, "data2/E-EX/Citrus/125041_s1/"),
            "DEM": "/home/tq/data2/citrus/hunan_data/hunan_DEM/125040_125041/",
            "Landsat_8": "/home/tq/data2/citrus/hunan_data/hunan_L8/125040_125041/",
        }, ],
        "work_path": "",
        "model_path": "",
        "traindata_path_npz": "",
        "field_name": "label",
        "res_label": "hn_125_Demo_0227",
        # ===========================
        "model_dir": "lstm_bn",
        "chunk": 4000,
        "year_date": ['2018', '0101', '1231'],
        # ===========================
        "shp_reproj_dict": {
            "samples":
                join(home_dir,
                     "data2/citrus/label/sichuan/sichuan_citrus_label_20190318.shp")
        },
        "read_order_list": "",
    }

    """
    state =
        0: get training data -> model training -> run prodict
        1: get training data -> model training
        2: get training data
        3: model training
    """
    source_path_list = glob.glob(
        join(
            home_dir,
            'tq-data03/TQLS_V0.1/tqls/*',
            process_dict['year_date'][0]
        )
    )

    source_tile_file = join(home_dir, 'data2/citrus/sc_sentinel_tile.json')

    with open(source_tile_file) as f:
        label_valid_tile = json.load(f)

    print(label_valid_tile)

    state = 2

    if state in [0, 1, 2]:
        for path in source_path_list:
            # make work path
            work_path = join(
                home_dir, "data2/citrus/demo/sample_result/TQLS/test/")

            tile = path.split('/')[-2]

            if tile not in label_valid_tile:
                continue
            else:
                print('PROCESS TILE ', tile)
                process_dict["work_path"] = join(work_path, "{}_{}_{}/".format(
                    socket.gethostname(), tile+'_tqls', time.strftime("%Y%m%dT%H%M%S")))

                if not os.path.exists(process_dict["work_path"]):
                    os.makedirs(process_dict["work_path"])
                process_dict['ori_ras_path']['DEM'] = join(path, 'DEM')
                process_dict['ori_ras_path']['Landsat_8'] = join(path, 'L30')
                process_dict['ori_ras_path']['Sentinel_2'] = join(path, 'S30')
                process_dict['ori_ras_path']['Sentinel_1'] = join(path, 'R30')
                control_tool(None, process_dict, process_label=state)
    if state == 3:
        j_file = join(
            home_dir, 'data2/citrus/demo/sample_result/TQLS/sichuan/merge_sample/traindata_merge_sample.json')
        control_tool(j_file, None, process_label=state)
        # TODO: data faltten for extracted files from different tiles
        # TODO: control handler for production only
