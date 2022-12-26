import os
import pandas as pd
import numpy as np

from utils.ipynb_helpers import read_data
from utils.stock_metrics import (
    LogPctProfitDirection,
    LogPctProfitTanh,
    apply_threshold_metric,
    pct_direction,
)
from utils.tools import dotdict


def open_results(log_dir, args, df):
    tpd_dict = {}
    for flag in ["train", "val", "test"]:
        device = 0
        while True:  # Device Loop
            preds_path = os.path.join(log_dir, f"results/pred_{flag}_{device}.npy")
            trues_path = os.path.join(log_dir, f"results/true_{flag}_{device}.npy")
            dates_path = os.path.join(log_dir, f"results/date_{flag}_{device}.npy")
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
                tpd_dict[flag] = (
                    dp
                    if flag not in tpd_dict
                    else [
                        np.append(tpdfi, dpi, axis=0)
                        for tpdfi, dpi in zip(tpd_dict[flag], dp)
                    ]
                )
                s = np.argsort(tpd_dict[flag][2], axis=None)
                tpd_dict[flag] = list(map(lambda x: x[s], tpd_dict[flag]))

                tpd_dict[flag][2] = pd.DatetimeIndex(tpd_dict[flag][2], tz="UTC")

                # Override trues with df target data to get original numerical precision
                if not ("mse" in args.loss and not args.inverse_output):
                    print("OVERRIDING trues with df target")
                    df_flag = df.loc[tpd_dict[flag][2]]
                    t = args.target.split("_")
                    df_target = df_flag[t[0]][t[1]].to_numpy()
                    tpd_dict[flag][0] = df_target

            else:
                # Done searching for devices
                break
            device += 1

    return tpd_dict


def get_metrics(
    args: dotdict, pred: np.ndarray, true: np.ndarray, thresh: float = 0.0
) -> dict:
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
    pct_profit_tanh = LogPctProfitTanh.metric(pred_f, true_f, short_filter=None)
    pct_profit_tanh_nshort = LogPctProfitTanh.metric(pred_f, true_f, short_filter="ns")
    pct_profit_tanh_oshort = LogPctProfitTanh.metric(pred_f, true_f, short_filter="os")

    # Optimal percent profit
    pct_profit_dir_opt = LogPctProfitDirection.metric(true_f, true_f)

    # Return metrics
    metrics = {
        "pct_profit_dir": pct_profit_dir,
        "pct_profit_dir_nshort": pct_profit_dir_nshort,
        "pct_profit_dir_oshort": pct_profit_dir_oshort,
        "pct_profit_tanh": pct_profit_tanh,
        "pct_profit_tanh_nshort": pct_profit_tanh_nshort,
        "pct_profit_tanh_oshort": pct_profit_tanh_oshort,
        "pct_excluded": (len(pred) - len(pred_f)) / len(pred),
        "pct_excluded_nshort": (len(pred) - len(pred_f[pred_f > 0])) / len(pred),
        "pct_excluded_oshort": (len(pred) - len(pred_f[pred_f < 0])) / len(pred),
        "pct_dir_correct": pct_dir_correct,
        "pct_profit_dir_opt": pct_profit_dir_opt,
    }
    return metrics
