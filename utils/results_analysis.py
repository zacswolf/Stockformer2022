import os
from pprint import pprint
from typing import Any
import pandas as pd
import numpy as np

from utils.stock_metrics import (
    LogPctProfitDirection,
    LogPctProfitTanhV1,
    LogPctProfitTanhV2,
    apply_threshold_metric,
    pct_direction,
)
from utils.tools import dotdict


def open_results(
    log_dir: str, args: dotdict, df: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Function to open an experiment and return its tpd_dict"""

    tpd_dict_tuple: dict[str, tuple[Any, Any, Any]] = {}

    for data_group in ["train", "val", "test"]:
        device = 0
        while True:  # Device Loop
            preds_path = os.path.join(
                log_dir, f"results/pred_{data_group}_{device}.npy"
            )
            trues_path = os.path.join(
                log_dir, f"results/true_{data_group}_{device}.npy"
            )
            dates_path = os.path.join(
                log_dir, f"results/date_{data_group}_{device}.npy"
            )
            if (
                os.path.exists(preds_path)
                and os.path.exists(trues_path)
                and os.path.exists(dates_path)
            ):
                dp = [
                    np.load(trues_path)[:, 0, 0],
                    np.load(preds_path)[:, 0, 0],
                    np.load(dates_path)[:, 0],
                ]
                tpd_dict_tuple[data_group] = (
                    dp
                    if data_group not in tpd_dict_tuple
                    else [
                        np.append(tpdfi, dpi, axis=0)
                        for tpdfi, dpi in zip(tpd_dict_tuple[data_group], dp)
                    ]
                )
                s = np.argsort(tpd_dict_tuple[data_group][2], axis=None)
                tpd_dict_tuple[data_group] = list(
                    map(lambda x: x[s], tpd_dict_tuple[data_group])
                )

                tpd_dict_tuple[data_group][2] = pd.DatetimeIndex(
                    tpd_dict_tuple[data_group][2], tz="UTC"
                )

                # Override trues with df target data to get original numerical precision
                if df is not None and not (
                    "mse" in args.loss and not args.inverse_output
                ):
                    print("OVERRIDING trues with df target")
                    df_data_group = df.loc[tpd_dict_tuple[data_group][2]]
                    t = args.target.split("_")
                    df_target = df_data_group[t[0]][t[1]].to_numpy()
                    tpd_dict_tuple[data_group][0] = df_target

            else:
                # Done searching for devices
                break
            device += 1

    tpd_dict: dict[str, dict[str, Any]] = {}
    for data_group in tpd_dict_tuple:
        tpd_dict[data_group] = {
            "trues": tpd_dict_tuple[data_group][0],
            "preds": tpd_dict_tuple[data_group][1],
            "dates": tpd_dict_tuple[data_group][2],
        }

    return tpd_dict


def get_metrics(
    args: dotdict | None, pred: np.ndarray, true: np.ndarray, thresh: float = 0.0
) -> dict:
    """Function to return metrics based on outputs and labels"""
    # Filter by a threshold
    pred_f, true_f = apply_threshold_metric(pred, true, thresh)
    # df_f = df.loc[date[np.abs(pred) >= thresh]]

    # Percent direction correct, ie up or down
    pct_dir_correct = pct_direction(pred_f, true_f)

    # Percent profit all in
    pct_profit_dir = LogPctProfitDirection.metric(pred_f, true_f, short_filter=None)
    pct_profit_dir_nshort = LogPctProfitDirection.metric(
        pred_f, true_f, short_filter="ns"
    )
    pct_profit_dir_oshort = LogPctProfitDirection.metric(
        pred_f, true_f, short_filter="os"
    )

    # Percent profit with tanh partial purchase
    pct_profit_tanhv1 = LogPctProfitTanhV1.metric(pred_f, true_f, short_filter=None)
    pct_profit_tanhv1_nshort = LogPctProfitTanhV1.metric(
        pred_f, true_f, short_filter="ns"
    )
    pct_profit_tanhv1_oshort = LogPctProfitTanhV1.metric(
        pred_f, true_f, short_filter="os"
    )

    pct_profit_tanhv2 = LogPctProfitTanhV2.metric(pred_f, true_f, short_filter=None)
    pct_profit_tanhv2_nshort = LogPctProfitTanhV2.metric(
        pred_f, true_f, short_filter="ns"
    )
    pct_profit_tanhv2_oshort = LogPctProfitTanhV2.metric(
        pred_f, true_f, short_filter="os"
    )

    # Optimal percent profit
    pct_profit_dir_opt = LogPctProfitDirection.metric(true_f, true_f)

    # Always 1 direction
    avg_pct_profit_always_short = np.power(
        LogPctProfitDirection.metric(-np.ones(pred_f.shape), true_f, short_filter=None),
        (1 / len(pred_f)),
    )
    avg_pct_profit_always_buy = np.power(
        LogPctProfitDirection.metric(np.ones(pred_f.shape), true_f, short_filter=None),
        (1 / len(pred_f)),
    )

    # Return metrics
    metrics = {
        "avg_pct_profit_tanhv1": np.power(pct_profit_tanhv1, (1 / len(pred_f))),
        "pct_profit_dir": pct_profit_dir,
        "pct_profit_dir_nshort": pct_profit_dir_nshort,
        "pct_profit_dir_oshort": pct_profit_dir_oshort,
        "pct_profit_tanhv1": pct_profit_tanhv1,
        "pct_profit_tanhv1_nshort": pct_profit_tanhv1_nshort,
        "pct_profit_tanhv1_oshort": pct_profit_tanhv1_oshort,
        "pct_profit_tanhv2": pct_profit_tanhv2,
        "pct_profit_tanhv2_nshort": pct_profit_tanhv2_nshort,
        "pct_profit_tanhv2_oshort": pct_profit_tanhv2_oshort,
        "pct_excluded": (len(pred) - len(pred_f)) / len(pred),
        "pct_excluded_nshort": (len(pred) - len(pred_f[pred_f > 0])) / len(pred),
        "pct_excluded_oshort": (len(pred) - len(pred_f[pred_f < 0])) / len(pred),
        "pct_dir_correct": pct_dir_correct,
        "pct_profit_dir_opt": pct_profit_dir_opt,
        "avg_pct_profit_always_short": avg_pct_profit_always_short,
        "avg_pct_profit_always_buy": avg_pct_profit_always_buy,
    }
    return metrics


def get_tuned_metrics(args: dotdict, tpd_dict: dict):
    # df = read_data(os.path.join(args.root_path, args.data_path))

    # Get the percentile to check thresh until
    what_percentile = 50
    percentile_value = 0.0

    for data_group in ["train"]:  # tpd_dict:
        preds = tpd_dict[data_group]["preds"]
        percentile_value += np.percentile(np.abs(preds), what_percentile)
    # percentile_value /= len(tpd_dict)
    print(f"{what_percentile}'th percentile: {percentile_value}")

    # Safety check
    ticker, field = args.target.split("_")
    assert field == "pctchange" or field == "logpctchange"

    # Tune threshhold based off of train's metric we care about
    if args.loss == "stock_tanhv1":
        tune_metric = "pct_profit_tanhv1"
    elif args.loss == "stock_tanhv2":
        tune_metric = "pct_profit_tanhv2"
    elif args.loss == "stock_tanhv4":
        tune_metric = "pct_profit_tanhv1"  # this is on purpose
    elif args.loss == "stock_tanh":
        tune_metric = "pct_profit_tanh"
    else:
        tune_metric = "pct_profit_dir"

    max_tracker = (0, 0)

    # Tracks results
    tracker = {}
    # Try all of the thresholds
    for thresh in np.linspace(0, percentile_value, 501):
        tracker[thresh] = {}

        for data_group in tpd_dict:
            true = tpd_dict[data_group]["trues"]
            pred = tpd_dict[data_group]["preds"]
            date = tpd_dict[data_group]["dates"]

            tracker[thresh][data_group] = get_metrics(args, pred, true, thresh)

            # Tune threshhold based off of train's metric we care about
            tune_value = tracker[thresh][data_group][tune_metric]
            if tune_value > max_tracker[0] and data_group == "train":
                max_tracker = (tune_value, thresh)

    best_thresh = max_tracker[1]

    print("zeros thresh:")
    for data_group in tracker[best_thresh]:
        print(data_group, end="\t")
        pprint(tracker[0.0][data_group], indent=3)
    print("\n\n")
    print("best thresh:", best_thresh)
    for data_group in tracker[best_thresh]:
        print(data_group, end="\t")
        pprint(tracker[best_thresh][data_group], indent=3)

    return best_thresh, tracker[best_thresh], tracker[0.0]
