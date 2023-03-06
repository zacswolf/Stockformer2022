"""
Bbtest stands for back-back test
This simulates training a model and testing it on some time frame and then training a new model on the timeframe after the previous, etc
Essentially its back testing not just our model, but our learning process several times along with the model each process produces
Example:
* Train k models, start with n months of data
* Model i in [1,k] gets trained on months [0:n+i-1], its validation data is month n+i, its test data is month n+i+1
"""
import argparse
from collections import defaultdict
import itertools
import json
from time import sleep
import yaml
import pickle
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from multiprocessing import current_process
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger

from run_once import pt_light_experiment
from utils.ipynb_helpers import read_data, bbtest_setting
from utils.results_analysis import get_tuned_metrics, open_results
from utils.tools import dotdict
from utils.parallel import NoDaemonProcessPool


LOG_BASE_DIR = "bbtest_logs"

# Each element can be passed to the device param during the pytorch lightning trainer initialization
GPU_LIST = list(map(lambda x: [x], range(2)))
NUM_PROC_PER_GPU = 1

NUM_TESTS: int | None = 8

NUM_PROC = len(GPU_LIST) * NUM_PROC_PER_GPU


def call_experiment(enumerated_args: list[tuple[int, dict, str, int]]):
    """Function to figure out what device to use and train a model based on that"""
    run_idx, args, setting, bbtest_id = enumerated_args
    args = dotdict(args)

    gpu_list_idx = (current_process()._identity[0] - 1) % len(GPU_LIST)

    logger = TensorBoardLogger(
        LOG_BASE_DIR, name=setting, flush_secs=15, version=run_idx
    )
    try:
        log_dir, test_loop_output = pt_light_experiment(
            args, devices=GPU_LIST[gpu_list_idx], logger=logger
        )
        assert logger.log_dir == log_dir
        return log_dir, dict(args), test_loop_output, bbtest_id, None
    except Exception as e:
        return None, dict(args), None, bbtest_id, e


def run_hypbbtest(
    config_file: str,
    test_duration: relativedelta,
    val_duration: relativedelta,
    data_start_date: datetime,
    data_end_date: datetime,
    test_window_start_date: datetime,
    hyper_params_changes: list = [],
):
    """Function to run potentially multiple back test on the learning algorithm. This is like a normal backtest except that we train a new model based off of `test_duration`."""
    # Open base config file
    with open(config_file, "r") as file:
        args_base = dotdict(yaml.full_load(file))

    if len(hyper_params_changes) == 0:
        hyper_params_changes = [{}]

    # We don't support multiple data sets atm
    df = read_data(os.path.join(args_base.root_path, args_base.data_path))

    hyp_analysis = {}

    # Get all the inputs for the bbtest
    inputs = []
    num_tests_remaining = {}
    full_test_dirs = []
    for bbtest_id, hp_override in enumerate(hyper_params_changes):
        sleep(5)
        args = dotdict(args_base | hp_override)
        assert args_base.data_path == args.data_path, "Data path must be the same"
        assert args_base.root_path == args.root_path, "Root path must be the same"
        actual_bb_inputs, setting = init_bbtest(
            test_duration,
            val_duration,
            data_start_date,
            data_end_date,
            test_window_start_date,
            bbtest_id,
            args,
        )
        full_test_dirs.append(os.path.join(LOG_BASE_DIR, setting))
        num_tests_remaining[bbtest_id] = len(actual_bb_inputs)
        inputs.extend(actual_bb_inputs)
        hyp_analysis[bbtest_id] = [hp_override, None]

    # Run the bbtests
    exceptions = []
    finished_bbtest_ids = []
    with NoDaemonProcessPool(processes=NUM_PROC) as pool:
        outputs = pool.map_async(call_experiment, inputs)

        # Open, Process, and Aggregate Test Data
        hyp_bb_tpd_dict = defaultdict(
            lambda: {
                "train": {"trues": [], "preds": [], "dates": []},
                "val": {"trues": [], "preds": [], "dates": []},
                "test": {"trues": [], "preds": [], "dates": []},
            }
        )
        for v_log_dir, args, test_loop_output, bbtest_id, e in outputs.get():
            args = dotdict(args)
            if e is None:
                # Open results from a version folder within the bbtest folder
                tpd_dict = open_results(v_log_dir, args, df)

                # Aggregate into bb_tpd_dict
                for data_group in ["train", "val", "test"]:
                    for key in ["trues", "preds", "dates"]:
                        hyp_bb_tpd_dict[bbtest_id][data_group][key].append(
                            tpd_dict[data_group][key]
                        )
            else:
                exceptions.append((e, bbtest_id))

            num_tests_remaining[bbtest_id] -= 1
            if num_tests_remaining[bbtest_id] == 0:
                # Finished all tests for this bbtest
                # Close out the bbtest
                print(hyp_bb_tpd_dict[bbtest_id], e)
                metrics = finish_bbtest(
                    full_test_dirs[bbtest_id], hyp_bb_tpd_dict[bbtest_id], args
                )
                hyp_analysis[bbtest_id][1] = metrics
                finished_bbtest_ids.append(bbtest_id)

                # Remove from dict to save memory
                del hyp_bb_tpd_dict[bbtest_id]

    if len(exceptions):
        print("Exceptions occured during bbtest:")
        for e, bbtest_id in exceptions:
            print(f"{full_test_dirs[bbtest_id]}:")
            print(e)

    # Make sure all bbtests are finished
    not_finished = list(
        set(range(len(hyper_params_changes))) - set(finished_bbtest_ids)
    )
    assert len(not_finished) == 0, f"bbtests {not_finished} did not finish"

    # # See what trends were the best across the different hyper params
    # # Rate each metric for each hyper param override
    # # Then plot the best hyper params for each metric
    # hyp_store = {}
    # for bbtest_id, (hp_override, metrics) in hyp_analysis.items():
    #     if metrics is None:
    #         continue
    #     for hyp, value in hp_override.items():
    #         if hyp not in hyp_store:
    #             hyp_store[hyp] = {}
    #         if value not in hyp_store[hyp]:
    #             hyp_store[hyp][value] = []
    #         hyp_store[hyp][value].append(metrics)

    # # Now we have a dict of dicts of lists of dicts
    # # Analyze hyp_store
    # hyp_mean = {}
    # for hyp, values in hyp_store.items():
    #     hyp_mean[hyp] = {}
    #     for value, metrics in values.items():
    #         hyp_mean[hyp][value] = sum(metrics) / len(metrics)  # calculate the mean

    # for bbtest_id, (hp_override, metrics) in hyp_analysis.items():
    #     if metrics is None:
    #         continue
    #     for hyp, value in hp_override.items():
    #         if hyp not in hyp_analysis:
    #             hyp_analysis[hyp] = {}
    #         if value not in hyp_analysis[hyp]:
    #             hyp_analysis[hyp][value] = []
    #         hyp_analysis[hyp][value].append(metrics)

    return None


def init_bbtest(
    test_duration,
    val_duration,
    data_start_date,
    data_end_date,
    test_window_start_date,
    bbtest_id,
    args,
):
    """Function to initialize a bbtest."""
    # Name this bbtest
    setting = bbtest_setting(args)
    print("Setting:", setting)

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
    assert len(bb_inputs) > 0, "No valid bbtest inputs, check your dates"

    # Add inputs to the list
    actual_bb_inputs = [
        (idx, args, setting, bbtest_id) for idx, args in enumerate(bb_inputs)
    ][(NUM_TESTS is not None) * -NUM_TESTS :]
    return actual_bb_inputs, setting


def finish_bbtest(bbtest_dir, bb_tpd_dict, args):
    """Function to finish a bbtest. This is where we calculate and save the metrics across all versions."""
    # Cast bb_tpd_dict to numpy arrays
    for data_group in ["train", "val", "test"]:
        # Check if any of the lists are empty
        if not all(
            len(bb_tpd_dict[data_group][val]) for val in ["trues", "preds", "dates"]
        ):
            return None

        bb_tpd_dict[data_group]["trues"] = np.concatenate(
            bb_tpd_dict[data_group]["trues"]
        )
        bb_tpd_dict[data_group]["preds"] = np.concatenate(
            bb_tpd_dict[data_group]["preds"]
        )

        bb_tpd_dict[data_group]["dates"] = bb_tpd_dict[data_group]["dates"][
            0
        ].union_many(bb_tpd_dict[data_group]["dates"][1:])

    with open(os.path.join(bbtest_dir, "tpd_dict.pickle"), "wb") as handle:
        pickle.dump(bb_tpd_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Get metrics
    best_thresh, best_thresh_metrics, zero_thresh_metrics = get_tuned_metrics(
        args, bb_tpd_dict
    )
    metrics = {
        0.0: zero_thresh_metrics,
        best_thresh: best_thresh_metrics,
    }
    with open(os.path.join(bbtest_dir, "metrics.json"), "w") as f:
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

    print("bbtest logged in:", bbtest_dir)
    return metrics


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="")
    bb_args = parser.parse_args()

    if bb_args.config_file == "":
        print("Using specified config file")
        config_file = "configs/stockformer/basic_WTH.yaml"
    else:
        config_file = bb_args.config_file

    # The duration of the test set, also the duration we slide with
    # test_duration = relativedelta(months=1)
    test_duration = relativedelta(months=1)

    # The duration of the val set
    val_duration = relativedelta(months=6)

    # OG NO COVID, oil
    # # Dataset bounds
    data_start_date = datetime.strptime("2012-01-01", "%Y-%m-%d")
    data_end_date = datetime.strptime("2020-01-01", "%Y-%m-%d")

    # # The date we should start the first testing window on
    test_window_start_date = datetime.strptime("2016-01-01", "%Y-%m-%d")

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
    # test_duration = relativedelta(weeks=1)
    # val_duration = relativedelta(weeks=6)
    # data_start_date = datetime.strptime("2017-01-01", "%Y-%m-%d")
    # data_end_date = datetime.strptime("2017-06-29", "%Y-%m-%d")
    # test_window_start_date = datetime.strptime("2017-04-14", "%Y-%m-%d")

    # switches = {
    #     "n_heads": [4, 6, 8],
    #     "dropout": [0.1, 0.25, 0.5],
    #     "e_layers": [3, 4, 6],
    #     "batch_size": [128, 256, 512],
    #     "seq_len": [16, 32, 64, 128],
    #     "learning_rate": [3.0e-4, 1.0e-4, 5.0e-5],
    # }

    # hyper_params_flat = [
    #     {key: value} for key in switches.keys() for value in switches[key]
    # ]

    # create dicts of all possible hyperparameter combinations in switches
    import itertools

    switches = {"n_heads": [4, 6, 8], "seq_len": [16, 32, 64, 128, 256]}

    # generate all possible combinations of hyperparameters using itertools.product()
    hyper_param_values = itertools.product(*switches.values())

    # create a list of hyperparameter dictionaries from the combinations
    hyper_params = [dict(zip(switches.keys(), values)) for values in hyper_param_values]

    hyper_params_changes = [
        {},
        {"n_heads": 12},
        {"dropout": 0.25},
        {"dropout": 0.1},
        {"dropout": 0.25, "n_heads": 12},
        {"dropout": 0.25, "e_layers": 4, "batch_size": 256},
        {"dropout": 0.25, "n_heads": 12, "e_layers": 4, "batch_size": 256},
        {"e_layers": 4, "batch_size": 256},
        {"n_heads": 12, "e_layers": 4, "batch_size": 256},
        {"seq_len": 128},
        {"seq_len": 128, "n_heads": 12},
        {"seq_len": 128, "dropout": 0.25},
        {"seq_len": 128, "dropout": 0.1},
        {"seq_len": 128, "dropout": 0.25, "n_heads": 12},
        {"seq_len": 128, "dropout": 0.25, "e_layers": 4},
        {"seq_len": 128, "dropout": 0.25, "n_heads": 12, "e_layers": 4},
        {"seq_len": 128, "e_layers": 4},
        {"seq_len": 128, "n_heads": 12, "e_layers": 4},
        # {"lr": 5e-5},
        # {"dropout": 0.75},
        # {"e_layers": 3},
        # {"e_layers": 3, "d_ff": 2048},
        # {"d_ff": 2048},
        # {"d_model": 256, "d_ff": 2048},
        # {"n_heads": 6},
        # {"n_heads": 4},
        # {"n_heads": 4, "e_layers": 3},
        # {"ln_mode": "pre"},
        # {"t_emb": "time2vec_add"},
        # {"t_emb": "time2vec_app"},
        # {"t_emb": "fixed"},
        # {"pre_loss": "stock_tanhv4", "pre_epochs": 10},
        # {"pre_loss": "mse", "pre_epochs": 10},
        # {"optim": "AdamW"},
        # {"lr": 1e-4},
        # {"seq_len": 32},
        # {"seq_len": 32, "num_heads": 6},
        # {"seq_len": 64},
        # {"seq_len": 128, "batch_size": 64},
        # {"seq_len": 256, "batch_size": 16},
        # {"seq_len": 256, "batch_size": 16, "num_heads": 16},
    ]
    run_hypbbtest(
        config_file,
        test_duration,
        val_duration,
        data_start_date,
        data_end_date,
        test_window_start_date,
        [{}],
    )
