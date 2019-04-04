import os
import gc
import sys
import math
import time
import logging
import numpy as np
from osgeo import gdal
# from sklearn.externals import joblib
import json
import glob

from baikal.general.feat_calc import feat_calc
from baikal.general.common import *
from baikal.ground_truth.TrainDataSG import TrainDataSG
from baikal.model.LSTM_predictor import TensorFlowPredictor

my_logger = logging.getLogger(__name__)


def load_json(json_file_path):
    with open(json_file_path, "r") as fp:
        tmp = json.load(fp)
    return tmp


def run_predictor(
    process_dict, *, start: int = 0, end: int = -1, coef: float = 10000
) -> (bool, str):
    """
    Function:
        Run saved model to produce classification results
    Input:
        process_dict, where useful paras are below:
            model_path: path stores the .m file
            pro_ras_list: list of raster dir where contains the raster bands to be
                        processed (for Landsat 8 data stored in seperate bands)
            work_path: stores the tmp files and final results.
            read_order_list: list of read order

        * optional parameters:
        start: start line
        end: end line
    Output:
        True for success!
        result stored in $work_path
    """
    # set para
    pro_ras_list = process_dict["pro_ras_list"]

    work_path = process_dict["work_path"]
    read_order_list = process_dict["read_order_list"]
    res_label = process_dict["res_label"]

    model_shortname = os.path.basename(process_dict["model_path"])
    model_label = model_shortname.split("_")[0]

    # load model
    # predictor = joblib.load(process_dict["model_path"])
    predictor = TensorFlowPredictor(process_dict["model_path"])
    # load training data
    traindata = np.load(process_dict["traindata_path_npy"])

    n_feat = traindata.shape[1] - 1

    # loop each file in list
    n_files = len(pro_ras_list)
    nf = 0
    for ras_file in pro_ras_list:
        nf += 1
        print("processing img set {}/{} :{}".format(nf, n_files, ras_file))

        # get band tiffs into a list
        bands_dict = get_bands_into_a_dict(ras_file, "*.tif")

        # read one input raster dataset
        ds = gdal.Open(next(bands_dict.walk())[-1])
        geo_trans = ds.GetGeoTransform()
        w = ds.RasterXSize
        h = ds.RasterYSize
        img_shape = [h, w]

        # prepare data
        split_num = 10
        predict_data_list = prepare_data(
            bands_dict, read_order_list, n_feat, work_path, split_num)
        if predict_data_list is None:
            my_logger.error("error in preparing data!")
            return False, None

        # loop to predict
        out_arr = np.zeros(img_shape).flatten()
        overall_offset_h = 0
        for predict_data_index, predict_data_path in enumerate(predict_data_list):
            # load predict subdata
            predict_sub_data = np.load(predict_data_path)
            print("load predict subdata file: {}".format(predict_data_path))

            # predict results of subdata
            assert predict_sub_data.shape[0] % w == 0, "predict subdata file error: {}".format(
                predict_data_path)
            sub_h = int(predict_sub_data.shape[0] / w)

            for dh in range(sub_h):
                sub_data = predict_sub_data[dh * w: dh * w + w, :]
                band = len(read_order_list)
                sub_data = sub_data.reshape((w, band, -1))
                t = predictor.predict_the_fuck(sub_data)
                # t = predictor.predict(predict_sub_data[dh * w: dh * w + w, :])
                out_arr[(overall_offset_h + dh) *
                        w: (overall_offset_h + dh) * w + w] = t
                print_progress_bar(dh + 1, sub_h)

            overall_offset_h = overall_offset_h + sub_h

        out_arr = out_arr.reshape(-1, w)
        print("predict done!")

        # delete predict data file
        for predict_data_path in predict_data_list:
            os.remove(predict_data_path)

        # build output path
        outpath = (
            work_path
            + model_label
            + "_result_"
            + time.strftime("%Y%m%d%H%M%S", time.localtime())
            + ".tif"
        )

        out_arr = out_arr.astype(np.int8)
        # write output into tiff file
        out_ds = gdal.GetDriverByName("GTiff").Create(
            outpath, w, h, 1, gdal.GDT_Byte)
        out_ds.SetProjection(ds.GetProjection())
        out_ds.SetGeoTransform(geo_trans)
        out_ds.GetRasterBand(1).WriteArray(out_arr)
        out_ds.FlushCache()
        my_logger.info("file write finished!")

    return True, outpath


def get_sub_array(raster_data_dic: dict, read_order_list: list, n_feat: int,
                  x_offset: int, y_offset: int, w: int, h: int):
    # get array from files in dictionary
    final_feature = dict()
    fn = 0
    # loop read order list, read all raster datas
    for ro in read_order_list:
        fn += 1
        print("{}/{} raster files:".format(fn, len(read_order_list)))
        source = ro[0]  # "Sentinel_1", "Landsat_8", "DEM"
        file_num = ro[1]  # "file_0", "file_1"
        ds = gdal.Open(raster_data_dic[source][file_num])
        n_bands = ds.RasterCount
        file_name = os.path.basename(raster_data_dic[source][file_num])
        time_str = file_name.split('_')[-8]
        final_feature[source].setdefault('time', []).append(time_str)

        for b in range(n_bands):
            bn = b + 1  # band_id, starts from 1
            print("-- {}/{} band:".format(bn, n_bands))
            data = ds.GetRasterBand(bn).ReadAsArray(
                x_offset, y_offset, w, h).reshape(-1, 1)
            if data is None:
                my_logger.error("get valid data error")
                return None, None

            my_logger.info(
                "getting data from ["
                + source
                + "] ["
                + file_num
                + "] {}".format(data.shape)
            )

            # append data to 2-D array: (points_num x times)
            band_keys = 'Band_' + str(bn)
            final_feature[source].setdefault(band_keys, []).append(data)
            print("    ->", final_feature[source][band_keys].shape)

    process_dict = single_run(iter=None, data_d=final_feature)

    res_array = []
    for key in process_dict.keys():
        data = process_dict[key]
        if not res_array:
            res_array = data
        else:
            res_array = np.concatenate((res_array, data), axis=1)
        print("     -> ", res_array.shape)

    # array = np.zeros([w * h, n_feat])
    # icol = 0
    # for ro in read_order_list:
    #     k1 = ro[0]
    #     k2 = ro[1]
    #     try:
    #         ds = gdal.Open(raster_data_dic[k1][k2])
    #         # print(raster_data_dic[k1][k2])
    #         for bi in range(ds.RasterCount):
    #             data = ds.GetRasterBand(
    #                 bi + 1).ReadAsArray(x_offset, y_offset, w, h).reshape(-1, 1)
    #             array[:, icol] = data.flatten()
    #             icol += 1
    #             # print(icol)
    #         ds = None
    #     except Exception as e:
    #         my_logger.error("open file error: {}".format(
    #             raster_data_dic[k1][k2]))
    #         return None
    #
    # array = replace_invalid_value(array, 0)

    return res_array


def prepare_data(raster_data_dic: dict, read_order_list: list, n_feat: int,
                 work_path: str, split_num: int):
    """
    Function:
        get ndarray from raster images in $raster_data_dic
    Input:
        raster_data_dic:  a 2-layered dict contains the raster files
                       as the model input.
                       like this:
                        {"sensor-1":{"band-1":"band path",
                                     "band-2":"band path",
                                    ...}
                         "sensor-2":{"band-1":"band path",
                                     "band-2":"band path",
                                    ...}
                        }
        read_order_list: list of read order
        n_feat: number of features
        split_num: number of the result sub_arrays
    Output:
        an ndarray contains all selected raster lines in shape of (:,1)
    """
    # open first raster file to get some infos
    try:
        first_raster_name = next(raster_data_dic.walk())[-1]
        ds = gdal.Open(first_raster_name)
        w = ds.RasterXSize
        h = ds.RasterYSize
        assert h >= split_num > 0, "split num error"
        h_index = [int(num * h / split_num) for num in range(split_num + 1)]
    except Exception:
        my_logger.error("open file error: %s", first_raster_name)
        return None

    # get array from files in dictionary
    predict_data_list = []
    for index in range(split_num):
        print("part{}".format(index + 1))
        x_offset = 0
        y_offset = h_index[index]
        sub_h = (h_index[index + 1] - h_index[index])
        sub_array = get_sub_array(raster_data_dic, read_order_list,
                                  n_feat, x_offset, y_offset, w, sub_h)
        predict_data_path = os.path.join(
            work_path, "predict_data_part_{}.npy".format(str(index+1)))
        np.save(predict_data_path, sub_array)
        predict_data_list.append(predict_data_path)
    return predict_data_list
