import sys
from utils.tools import dotdict
from exp.exp_informer import Exp_Informer
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.ipynb_helpers import args_from_setting, setting_from_args, handle_gpu, read_data


args = dotdict()



args.des = 'full_1h'
args.model = 'stockformer' # 'stockformer'
args.data = 'custom' # data
args.checkpoints = './checkpoints' # location of model checkpoints
args.root_path = './data/ETT/' # root path of data file
args.data_path = 'full_1h.csv' # data file
args.freq = 'h' # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
args.features = 'MS' # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = 'XOM_pctchange' # target feature in S or MS task
args.label_len = 1 # start token length of Informer decoder
args.pred_len = 1 # prediction sequence length
args.cols = ["XOM_pctchange", #"XOM_open", "XOM_close", , "XOM_shortsma", 
                'CVX_pctchange', 'COP_pctchange', 'BP_pctchange', 'PBR_pctchange', 
                'WTI_pctchange', 'EOG_pctchange', 'ENB_pctchange', 'SLB_pctchange',
                ]#'C:USDSAR_pctchange'
args.enc_in = len(args.cols) # encoder input size
# args.dec_in = len(args.cols) # decoder input size # TODO: Remove
args.c_out = 1 if args.features in ["S", "MS"] else args.dec_in # output size

args.embed = None#'timeF' # time features encoding, options:[timeF, fixed, learned, None]
args.activation = 'gelu' # activation

args.attn = 'full' # attention used in encoder, options:[prob, full]
args.factor = 5 # probsparse attn factor; doesn't matter unless args.attn==prob
args.distil = False # whether to use distilling in encoder
args.output_attention = False # whether to output attention in encoder
args.mix = False # whether to use mixed attention
args.padding = 0 # TODO: Remove

args.train_epochs = 50
args.patience = 30 # For early stopping

args.use_amp = False # whether to use automatic mixed precision training
args.num_workers = 0
args.itr = 1 # number of runs

args.scale = True # whether to scale to mean 0, var 1
args.inverse = True # whether to invert that scale before loss is calculated, lets keep this at False

# This is for debugging to overfit
# When True, patience doesn't matter at all and the model-state that is saved is the one after the last epoch
# When False, the model-state that is saved is the one with the highest validation-loss and we can early stop with patience
args.no_early_stop = False 


# Control data split from args, either a date string like "2000-01-30" or None (for default)
args.date_start = "2012-01-01" # Train data starts on this date, default is to go back as far as possible
args.date_end = "2020-01-01" # Train data starts on this date, default is to go back as far as possible
args.date_test = "2019-06-01" # Test data is data after this date, default is to use ~20% of the data as test data
args.learning_rate = 0.0001

# =================================================

seq_len_list = [16, 20, 24, 28, 32, 36, 50, 64, 128]
d_model_list = [128, 256, 512, 1024]
n_heads_list = [4, 8, 12, 16, 32, 64]
e_layers_list = [8, 12, 24, 48]
d_ff_list = [1024, 2048, 3072]
dropout_list = [0.05, 0.25, 0.5]
batch_size_list = [64, 128, 256]
loss_list = ["mse", "stock_lpp", "stock_lppws", "stock_tanh"]
lradj_list = [None, "type3", "type1"]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 

# args.seq_len = 16 # input sequence length of Informer encoder
# args.d_model = 128 # dimension of model; this is also the dimension of the token embeddings
# args.n_heads = 8 # num of attention heads
# args.e_layers = 12 # num of encoder layers
# # args.d_layers = 4 # num of decoder layers # TODO: Remove
# args.d_ff = 2048 # dimension of fcn in model
# args.dropout = 0.05 # dropout
# args.batch_size = 256 #64
# args.loss = 'stock_lpp' # What loss function to use: ["mse", "stock_lpp", "stock_lppws", "stock_tanh"]
# args.lradj = "type3"#"type3" # What learning rate scheduler to use: ["type3", None, "type1"]

# -=-==-=-==-==-=

def train_loop(args):
    exp = None
    setting = None
    for ii in range(args.itr):
        # setting record of experiments
        setting = setting_from_args(args, ii)
        
        # set experiments
        exp = Exp(args)
        
        # train
        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)

        # test
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, flag="test", inverse=True)
        exp.test(setting, flag="val", inverse=True)
        exp.test(setting, flag="train", inverse=True)
        torch.cuda.empty_cache()
    
    manual = False
    if manual:
        setting = "stockformer_custom_ftMS_sl16_ll4_pl1_ei12_di12_co1_iFalse_dm512_nh8_el12_dl4_df2048_atfull_fc5_ebNone_dtFalse_mxFalse_full_1h_0"
        args = args_from_setting(setting, args)
        exp = Exp(args)
    path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    exp.predict(setting, True)
    # the prediction will be saved in ./results/{setting}/real_prediction.npy
    prediction = np.load(f"./results/{setting}/real_prediction.npy")

    tp_dict = {}
    for flag in ["train", "val", "test"]:
        preds_path = f"./results/{setting}/pred_{flag}.npy"
        trues_path = f"./results/{setting}/true_{flag}.npy"
        dates_path = f"./results/{setting}/date_{flag}.npy"
        if os.path.exists(preds_path) and os.path.exists(trues_path) and os.path.exists(dates_path):
            tp_dict[flag] = (np.load(trues_path), np.load(preds_path), np.load(dates_path))
            # tp_dict[flag] = list(zip(*sorted(zip(*tp_dict[flag]), key=lambda x: x[-1])))
            s = np.argsort(tp_dict[flag][2], axis=None)
            tp_dict[flag] = list(map(lambda x: x[s], tp_dict[flag]))
    
    max_tracker = (0, 0)

    # Tracks results
    tracker = {}

    df = read_data(os.path.join(args.root_path, args.data_path))

    # Get the percentile to check thresh until
    percentile = [50, 0.0]
    for flag in ["train"]: # tp_dict:
        _, preds, _ = tp_dict[flag]
        percentile[1] += np.percentile(np.abs(preds), percentile[0]) #np.median(np.abs(preds))
    percentile[1] /= len(tp_dict)
    print(f"{percentile[0]}'th percentile: {percentile[1]}")

    ticker, field  = args.target.split("_")
    assert field == "pctchange"

    for thresh in np.linspace(0, .00025, 501):
        # print("thresh:", thresh)
        tracker[thresh] = {}
        track = {}
        for flag in tp_dict:
            trues, preds, dates = tp_dict[flag]
            # trues, preds = np.exp(trues), np.exp(preds)
            true = trues[:,0,0].copy()
            pred = preds[:,0,0].copy()
            date = pd.DatetimeIndex(dates[:,0], tz="UTC")

            df_flag = df.loc[date][np.abs(pred) >= thresh]
            
            # Filter by thresh. Note in log scale
            true_c_log = true[np.abs(pred) >= thresh]
            pred_c_log = pred[np.abs(pred) >= thresh]

            # Percent direction correct, ie up or down
            pct_dir_correct = np.sum(np.sign(true_c_log) == np.sign(pred_c_log))/len(true_c_log)

            true_c, pred_c = np.exp(true_c_log), np.exp(pred_c_log)

            # # Turn pct_change to price change
            # true_price_change = df_flag[ticker]["open"] * (true_c-1)
            # pred_price_change = df_flag[ticker]["open"] * (pred_c-1)
            # # Profit if you always bought one share with shorting
            # p_one_share_wshort = (true_price_change * np.sign(pred_price_change)).sum()
            # # Profit if you always bought one share without shorting
            # p_one_share = (true_price_change * np.sign(pred_price_change))[pred_price_change > 0].sum()


            # Important: Percent profit with & without shorting
            # pct_profit_wshort = ((true_c-1) * np.sign(pred_c-1) + 1).prod()
            pct_profit_wshort = np.exp((true_c_log * np.sign(pred_c_log)).sum())
            # pct_profit = ((true_c-1) * np.sign(pred_c-1) + 1)[pred_c > 1].prod()
            pct_profit = np.exp((true_c_log * np.sign(pred_c_log))[pred_c_log > 0].sum())

            # Important: percent profit with & without shorting with partial purchase
            pct_profit_tanh_wshort = np.exp((true_c_log * np.tanh(1000*pred_c_log)).sum())
            pct_profit_tanh = np.exp((true_c_log * np.tanh(1000*pred_c_log))[pred_c_log > 0].sum())

            # Optimal percent profit without shorting
            # pct_profit_opt = ((true_c-1) * np.sign(true_c-1) + 1)[true_c > 1].prod()
            pct_profit_opt = np.exp((true_c_log * np.sign(true_c_log))[true_c_log > 0].sum())

            # Tune threshhold based off of train's metric we care about
            tune_metric = pct_profit_tanh if args.loss == "stock_tanh" else pct_profit
            if tune_metric > max_tracker[0] and flag=='train':
                max_tracker = (tune_metric, thresh)

            # Save
            tracker[thresh][flag] = {
                "pct_profit": pct_profit, "pct_profit_wshort": pct_profit_wshort, 
                # "p_one_share": p_one_share, "p_one_share_wshort": p_one_share_wshort, 
                "pct_profit_tanh": pct_profit_tanh, "pct_profit_tanh_wshort": pct_profit_tanh_wshort,
                "pct_excluded": (len(pred) - len(pred_c_log[pred_c_log > 0]))/len(pred),
                "pct_excluded_wshort": (len(pred) - len(pred_c_log))/len(pred),
                "pct_dir_correct": pct_dir_correct,
                "pct_profit_opt": pct_profit_opt
            }
    best_thresh = max_tracker[1]
    print("best thresh:", best_thresh)
    val_profit = tracker[best_thresh]["val"]["pct_profit"]
    return val_profit, best_thresh
    # for k in tracker[best_thresh]:
    #     print(f"{k}\t", tracker[best_thresh][k])


best_val_thresh_args = [0,0,0]
for seq_len in seq_len_list:
    for d_model in d_model_list:
        for n_heads in n_heads_list:
            for e_layers in e_layers_list:
                for d_ff in d_ff_list:
                    for dropout in dropout_list:
                        for batch_size in batch_size_list:
                            for loss in loss_list:
                                for lradj in lradj_list:
                                    args.seq_len = seq_len
                                    args.d_model = d_model
                                    args.n_heads = n_heads
                                    args.e_layers = e_layers
                                    args.d_ff = d_ff
                                    args.dropout = dropout
                                    args.batch_size = batch_size
                                    args.loss = loss
                                    args.lradj = lradj
                                    handle_gpu(args, "4,5,6,7")
                                    args.detail_freq = args.freq
                                    args.freq = args.freq[-1:]
                                    print('Args in experiment:')
                                    for key in args.keys():
                                        print(key,", ",args[key])
                                    Exp = Exp_Informer
                                    val_profit, best_thresh = train_loop(args)
                                    if val_profit > best_val_thresh_args[0]:
                                        best_val_thresh_args[0] = val_profit
                                        best_val_thresh_args[1] = best_thresh
                                        best_val_thresh_args[2] = args
                                        print("###############################")
                                        print("best so far:")
                                        print("val profit: ",val_profit)
                                        print("val best thresh: ",best_thresh)
                                        for key in args.keys():
                                            print(key,", ",args[key])
                                        print("###############################")
                                        print()
                                        print()


print("###################################")
print("best for all")
print("###############################")
print("val profit: ",best_val_thresh_args[0])
print("val best thresh: ",best_val_thresh_args[1])
for key in best_val_thresh_args[2].keys():
    print(key,", ",best_val_thresh_args[2][key])
print("###############################")