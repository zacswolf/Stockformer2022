import datetime
import json
import os
import re

import torch
from utils.tools import dotdict

import pandas as pd


# Args / Settings helper functions


def args_from_setting(setting, args):
    # pattern = r"(.+)_(.+)_ft(.+)_sl(.+)_ll(.+)_pl(.+)_ei(.+)_di(.+)_co(.+)_i(.+)_dm(.+)_nh(.+)_el(.+)_dl(.+)_df(.+)_at(.+)_fc(.+)_eb(.+)_dt(.+)_mx(.+)_(.+)_(.+).*"
    # match = re.search(pattern, setting)
    # if match:
    #     conv = lambda x: int(x) if x.isdigit() else (False if x=="False" else (True if x=="True" else x))

    #     (args.model, args.data, args.features,
    #     args.seq_len, args.label_len, args.pred_len,
    #     args.enc_in, args.dec_in, args.c_out, args.inverse,
    #     args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor,
    #     args.embed, args.distil, args.mix, args.des, ii) = map(conv, match.groups())
    #     print(args)
    # else:
    #     raise Exception("Issue with setting name")
    path = f"results/{setting}/args.json"
    assert os.path.exists(path), f"{path}/args.json doesn't exist"

    with open(path, "r") as f:
        args = json.load(f)
        return dotdict(args)


def setting_from_args(args, ii=0):
    setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_ei{}_di{}_co{}_i{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}".format(
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.enc_in,
        args.dec_in,
        args.c_out,
        args.inverse,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.attn,
        args.factor,
        args.embed,
        args.distil,
        args.mix,
        args.des,
        ii,
    )

    return setting


def write_df(data, out_file, append=""):
    # Save flatten
    og_cols = data.columns.copy()
    data.columns = data.columns.to_flat_index()

    data.columns = pd.Index(["_".join(col) for col in data.columns])

    if append:
        dot_loc = out_file.rfind(".")
        out_file = f"{out_file[:dot_loc]}_{append}{out_file[dot_loc:]}"

    if os.path.exists(out_file):
        # Move current file to data/old
        data_old = "data/old"
        if not os.path.exists(data_old):
            os.makedirs(data_old)
        new_file_name = f"{out_file[:out_file.rfind('.')].replace('./','').replace('/','_')}_{datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}{out_file[out_file.rfind('.'):]}"
        os.rename(out_file, os.path.join(data_old, new_file_name))

    data.to_csv(out_file)
    data.columns = og_cols
    return out_file


# write_df(df, "test.csv")
def read_data(out_file="realdata.csv"):
    data = pd.read_csv(out_file, index_col=0)

    converter = lambda col: tuple(col.split("_"))
    # ast.literal_eval
    data.columns = data.columns.map(converter)
    data.index = pd.to_datetime(data.index)
    return data


def add_tz(data, time_zone="US/Eastern"):
    """Add timezone to timezone-unlabled df"""
    t = pd.to_datetime(data.index).to_series()
    data.index = t.dt.tz_localize(time_zone)
    return data


def convert_tz(data, time_zone="US/Eastern"):
    t = data.index.to_series()
    t = t.dt.tz_convert(time_zone)
    data.index = t
    return data


# args.use_gpu = True if torch.cuda.is_available() else False
# args.gpu = 1

# args.use_multi_gpu = True
# args.devices = '0,1'
# if args.use_gpu and args.use_multi_gpu:
#     args.devices = args.devices.replace(' ','')
#     device_ids = args.devices.split(',')
#     args.device_ids = [int(id_) for id_ in device_ids]
#     args.gpu = args.device_ids[0]
def handle_gpu(args, gpu=None):
    if not gpu and gpu is not None:
        # Don't use gpu
        args.use_gpu = False
        args.use_multi_gpu = False
        return

    args.use_gpu = True if torch.cuda.is_available() else False

    if not args.use_gpu:
        return

    if gpu is None:
        # Use all gpus
        c = torch.cuda.device_count()

        args.device_ids = list(map(int, range(torch.cuda.device_count())))
        args.devices = ",".join(map(str, args.device_ids))
    else:
        # Passed gpu(s)
        gpu = str(gpu)

        args.devices = gpu.replace(" ", "")
        args.device_ids = [int(id_) for id_ in args.devices.split(",")]

    if len(args.device_ids) >= 1:
        args.use_multi_gpu = len(args.device_ids) > 1
        args.gpu = int(args.device_ids[0])
