"""
Bbtest stands for back-back test
This simulates training a model and testing it on some time frame and then training a new model on the timeframe after the previous, etc
Essentially its back testing not just our model, but our learning process several times along with the model each process produces
Example:
* Train k models, start with n months of data
* Model i in [1,k] gets trained on months [0:n+i-1], its validation data is month n+i, its test data is month n+i+1
"""
from collections import defaultdict
import json
from time import sleep
import yaml
import pickle
import os

print("0")
from pprint import pprint
from datetime import datetime
from dateutil.relativedelta import relativedelta
from multiprocessing import current_process
import numpy as np

print("A")
from pytorch_lightning.loggers import TensorBoardLogger

print("B")

from run_once import pt_light_experiment
from utils.ipynb_helpers import read_data, bbtest_setting
from utils.results_analysis import get_tuned_metrics, open_results
from utils.tools import dotdict
from utils.parallel import NoDaemonProcessPool


LOG_BASE_DIR = "bbtest_logs"

# Each element can be passed to the device param during the pytorch lightning trainer initialization
GPU_LIST = list(map(lambda x: [x], range(8)))


def call_experiment(enumerated_args: list[tuple[int, dict, str, int]]):
    """Function to figure out what device to use and train a model based on that"""
    run_idx, args, setting, bbtest_id = enumerated_args
    args = dotdict(args)

    gpu_list_idx = current_process()._identity[0] - 1

    logger = TensorBoardLogger(
        LOG_BASE_DIR, name=setting, flush_secs=15, version=run_idx
    )
    log_dir, test_loop_output = pt_light_experiment(
        args, devices=GPU_LIST[gpu_list_idx], logger=logger, save_metrics=1
    )
    assert logger.log_dir == log_dir
    return log_dir, dict(args), test_loop_output, bbtest_id


def run_bbtest(
    config_file: str,
    test_duration: relativedelta,
    val_duration: relativedelta,
    data_start_date: datetime,
    data_end_date: datetime,
    test_window_start_date: datetime,
    hyper_params_changes: list = [],
):
    """Function to run a back test on the learning algorithm. This is like a normal backtest except that we train a new model based off of `test_duration`."""
    # Open base config file
    with open(config_file, "r") as file:
        args_base = dotdict(yaml.full_load(file))

    if len(hyper_params_changes) == 0:
        hyper_params_changes = [{}]

    inputs = []
    full_test_dirs = []
    for bbtest_id, hp_override in enumerate(hyper_params_changes):
        sleep(5)
        args = dotdict(args_base | hp_override)
        # Name this bbtest
        setting = bbtest_setting(args)
        print("Setting:", setting)
        full_test_dirs.append(os.path.join(LOG_BASE_DIR, setting))

        # Create input list for multi process
        bb_inputs = []
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
                bb_inputs.append(dict(args))

        # NOTE: the [-8:] should technically not be used here for a true bbtest
        # However, just having 1 batch of runs is way faster
        bb_inputs = [
            (idx, args, setting, bbtest_id) for idx, args in enumerate(bb_inputs)
        ][-len(GPU_LIST) :]
        inputs.extend(bb_inputs)

    # We don't support multiple data sets atm
    df = read_data(os.path.join(args.root_path, args.data_path))

    with NoDaemonProcessPool(processes=len(GPU_LIST)) as pool:
        outputs = pool.map_async(call_experiment, inputs)

        # Open, Process, and Aggregate Test Data
        bb_tpd_dict = defaultdict(
            lambda: {
                "train": {"trues": [], "preds": [], "dates": []},
                "val": {"trues": [], "preds": [], "dates": []},
                "test": {"trues": [], "preds": [], "dates": []},
            }
        )
        test_loop_outputs = []
        for log_dir, args, test_loop_output, bb_test_id in outputs.get():
            args = dotdict(args)
            test_loop_outputs.append(test_loop_output)

            for data_group in ["train", "val", "test"]:
                tpd_dict = open_results(log_dir, args, df)

                true = tpd_dict[data_group]["trues"]
                pred = tpd_dict[data_group]["preds"]
                date = tpd_dict[data_group]["dates"]
                bb_tpd_dict[bb_test_id][data_group]["trues"].append(true)
                bb_tpd_dict[bb_test_id][data_group]["preds"].append(pred)
                bb_tpd_dict[bb_test_id][data_group]["dates"].append(date)

    # Aggregate and cast
    for bb_test_id in bb_tpd_dict.keys():
        for data_group in ["train", "val", "test"]:
            bb_tpd_dict[bb_test_id][data_group]["trues"] = np.concatenate(
                bb_tpd_dict[bb_test_id][data_group]["trues"]
            )
            bb_tpd_dict[bb_test_id][data_group]["preds"] = np.concatenate(
                bb_tpd_dict[bb_test_id][data_group]["preds"]
            )
            bb_tpd_dict[bb_test_id][data_group]["dates"] = bb_tpd_dict[bb_test_id][
                data_group
            ]["dates"][0].union_many(bb_tpd_dict[bb_test_id][data_group]["dates"][1:])

        with open(
            os.path.join(full_test_dirs[bb_test_id], "tpd_dict.pickle"), "wb"
        ) as handle:
            pickle.dump(
                bb_tpd_dict[bb_test_id], handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    #### Analyze
    for bb_test_id in bb_tpd_dict.keys():
        best_thresh, best_thresh_metrics, zero_thresh_metrics = get_tuned_metrics(
            args, bb_tpd_dict[bb_test_id]
        )
        metrics = {0.0: zero_thresh_metrics, best_thresh: best_thresh_metrics}
        with open(os.path.join(full_test_dirs[bb_test_id], "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Warnings
        action_diff = np.abs(
            metrics[0.0]["test"]["pct_excluded_nshort"]
            - metrics[0.0]["test"]["pct_excluded_oshort"]
        )
        if action_diff > 0.6:
            print("WARNING: significant action preference between buying shorting")

        train_pct_dir_correct = metrics[0.0]["train"]["pct_dir_correct"]
        if train_pct_dir_correct < 0.55:
            print(
                f"WARNING: train isn't properly learning direction. pct_dir_correct: {train_pct_dir_correct}"
            )

        print("bbtest logged in:", full_test_dirs[bb_test_id])
    return None


if __name__ == "__main__":
    config_file = "configs/lstm/basic_PEMSBAY.yaml"

    # The duration of the test set, also the duration we slide with
    # test_duration = relativedelta(months=1)
    test_duration = relativedelta(months=1)

    # The duration of the val set
    # val_duration = relativedelta(weeks=6)  # months=6)
    val_duration = relativedelta(months=6)

    # OG NO COVID, oil
    # # Dataset bounds
    # data_start_date = datetime.strptime("2012-01-01", "%Y-%m-%d")
    # data_end_date = datetime.strptime("2020-01-01", "%Y-%m-%d")

    # # The date we should start the first testing window on
    # test_window_start_date = datetime.strptime("2016-01-01", "%Y-%m-%d")

    # Messing around, oil
    # test_duration = relativedelta(months=2)
    # val_duration = relativedelta(months=1)
    # data_start_date = datetime.strptime("2012-01-01", "%Y-%m-%d")
    # data_end_date = datetime.strptime("2022-11-10", "%Y-%m-%d")
    # test_window_start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")

    # WTH
    # test_duration = relativedelta(months=1)
    # val_duration = relativedelta(months=6)
    # data_start_date = datetime.strptime("2010-01-01", "%Y-%m-%d")
    # data_end_date = datetime.strptime("2013-12-01", "%Y-%m-%d")
    # test_window_start_date = datetime.strptime("2013-05-01", "%Y-%m-%d")

    # PEMSBAY
    test_duration = relativedelta(weeks=1)
    val_duration = relativedelta(weeks=6)
    data_start_date = datetime.strptime("2017-01-01", "%Y-%m-%d")
    data_end_date = datetime.strptime("2017-06-29", "%Y-%m-%d")
    test_window_start_date = datetime.strptime("2017-04-14", "%Y-%m-%d")

    hyper_params_changes = [
        # {"e_layers": 1},
        # {"learning_rate": 5.0e-5},
        # {"dropout": 0.5},
        # {"d_model": 512},
        # {"d_ff": 512},
        {},
        {"lradj": "type2"},
        {"no_scale_mean": False},
        {"max_epochs": 65, "pre_loss": "stock_tanhv4", "pre_epochs": 15},
        {"max_epochs": 80},
        {"seq_len": 32},
    ]

    run_bbtest(
        config_file,
        test_duration,
        val_duration,
        data_start_date,
        data_end_date,
        test_window_start_date,
        hyper_params_changes,
    )
