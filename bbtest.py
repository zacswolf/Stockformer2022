import json
import yaml
import pickle
import os
from pprint import pprint
from datetime import datetime
from dateutil.relativedelta import relativedelta
from multiprocessing import current_process
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

from pt_light import pt_light_expiriment
from utils.ipynb_helpers import read_data, bbtest_setting
from utils.results_analysis import get_tuned_metrics, open_results
from utils.tools import dotdict
from utils.parallel import NoDaemonProcessPool


LOG_BASE_DIR = "bbtest_logs"

# Each element can be passed to the device param during the pytorch lightning trainer initialization
GPU_LIST = list(map(lambda x: [x], range(8)))


def call_expiriment(enumerated_args: list[tuple[int, dict, str]]):
    """Function to figure out what device to use and train a model based on that"""
    run_idx, args, setting = enumerated_args
    args = dotdict(args)

    gpu_list_idx = current_process()._identity[0] - 1

    logger = TensorBoardLogger(
        LOG_BASE_DIR, name=setting, flush_secs=15, version=run_idx
    )
    log_dir, test_loop_output = pt_light_expiriment(
        args, devices=GPU_LIST[gpu_list_idx], logger=logger
    )
    assert logger.log_dir == log_dir
    return log_dir, dict(args), test_loop_output


def run_bbtest(
    config_file: str,
    test_duration: relativedelta,
    val_duration: relativedelta,
    data_start_date: datetime,
    data_end_date: datetime,
    test_window_start_date: datetime,
):
    """Function to run a back test on the learning algorithm. This is like a normal backtest except that we train a new model based off of `test_duration`."""
    # Open base config file
    with open(config_file, "r") as file:
        args = dotdict(yaml.full_load(file))

    # Name this bbtest
    setting = bbtest_setting(args)
    print("Setting:", setting)
    full_test_dir = os.path.join(LOG_BASE_DIR, setting)

    # Create input list for multi process
    inputs = []
    date_end = test_window_start_date
    done = False
    while not done:
        # Change args
        args.date_start = data_start_date.strftime("%Y-%m-%d")
        args.date_test = date_end.strftime("%Y-%m-%d")
        args.date_val = (date_end - val_duration).strftime("%Y-%m-%d")
        date_end = date_end + test_duration
        args.date_end = date_end.strftime("%Y-%m-%d")

        if date_end > data_end_date:
            done = True
        else:
            inputs.append(dict(args))

    # NOTE: the [-8:] should technically not be used here for a true bbtest
    # However, just having 1 batch of runs is way faster
    inputs = [(idx, args, setting) for idx, args in enumerate(inputs)][-8:]

    df = read_data(os.path.join(args.root_path, args.data_path))

    with NoDaemonProcessPool(processes=len(GPU_LIST)) as pool:
        outputs = pool.map_async(call_expiriment, inputs)

        # Open, Process, and Aggregate Test Data
        bb_tpd_dict = {
            "train": {"trues": [], "preds": [], "dates": []},
            "val": {"trues": [], "preds": [], "dates": []},
            "test": {"trues": [], "preds": [], "dates": []},
        }
        for log_dir, args, test_loop_output in outputs.get():
            args = dotdict(args)

            test_loop_outputs = []
    for data_group in ["train", "val", "test"]:
            tpd_dict = open_results(log_dir, args, df)

            true = tpd_dict[data_group]["trues"]
            pred = tpd_dict[data_group]["preds"]
            date = tpd_dict[data_group]["dates"]
                bb_tpd_dict[data_group]["trues"].append(true)
                bb_tpd_dict[data_group]["preds"].append(pred)
                bb_tpd_dict[data_group]["dates"].append(date)
            test_loop_outputs.append(test_loop_output)

    # Aggregate and cast
    for data_group in ["train", "val", "test"]:
        bb_tpd_dict[data_group]["trues"] = np.concatenate(
            bb_tpd_dict[data_group]["trues"]
        )
        bb_tpd_dict[data_group]["preds"] = np.concatenate(
            bb_tpd_dict[data_group]["preds"]
        )
        bb_tpd_dict[data_group]["dates"] = bb_tpd_dict[data_group]["dates"][
            0
        ].union_many(bb_tpd_dict[data_group]["dates"][1:])

    with open(os.path.join(full_test_dir, "tpd_dict.pickle"), "wb") as handle:
        pickle.dump(bb_tpd_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #### Analyze

    best_thresh, best_thresh_metrics, zero_thresh_metrics = get_tuned_metrics(
        args, bb_tpd_dict
    )
    metrics = {0.0: zero_thresh_metrics, best_thresh: best_thresh_metrics}
    with open(os.path.join(full_test_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("bbtest logged in:", full_test_dir)
    return metrics


if __name__ == "__main__":
    config_file = "configs/stockformer/general.yaml"

    # The duration of the test set, also the duration we slide with
    test_duration = relativedelta(months=1)

    # The duration of the val set
    val_duration = relativedelta(months=6)

    # Full dataset bounds
    data_start_date = datetime.strptime("2012-01-01", "%Y-%m-%d")
    data_end_date = datetime.strptime("2020-01-01", "%Y-%m-%d")

    # The date we should start the first testing window on
    test_window_start_date = datetime.strptime("2016-01-01", "%Y-%m-%d")

    run_bbtest(
        config_file,
        test_duration,
        val_duration,
        data_start_date,
        data_end_date,
        test_window_start_date,
    )
