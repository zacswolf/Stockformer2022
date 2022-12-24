import os
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    StochasticWeightAveraging,
    LearningRateMonitor,
)
from utils.callbacks import PredTrueDateWriter
from pytorch_lightning.loggers import TensorBoardLogger
from data_provider.data_module import CustomDataModule
from exp.exp_timeseries import ExpTimeseries
from utils.tools import dotdict

from utils.ipynb_helpers import (
    setting_from_args,
    handle_gpu,
)
import pytorch_lightning as pl


args = dotdict()
args.des = "full_1h"

args.model = "stockformer"  # 'stockformer'

args.data = "custom"  # data
args.checkpoints = "./checkpoints"  # location of model checkpoints
args.root_path = "./data/stock/"  # root path of data file

args.data_path = "full_1h.csv"  # data file
args.freq = "h"  # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h

args.features = "MS"  # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
args.target = "WTI_logpctchange"  # target feature in S or MS task


args.seq_len = 16  # input sequence length of Informer encoder
args.label_len = 0  # start token length of Informer decoder
args.pred_len = 1  # prediction sequence length

args.cols = [
    "XOM_logpctchange",
    "CVX_logpctchange",
    "COP_logpctchange",
    "BP_logpctchange",
    "PBR_logpctchange",
    "WTI_logpctchange",
    "EOG_logpctchange",
    "ENB_logpctchange",
    "SLB_logpctchange",
]  #'C:USDSAR_logpctchange'

args.enc_in = len(args.cols)  # encoder input size
# args.dec_in = len(args.cols) # decoder input size # TODO: Remove
args.c_out = 1 if args.features in ["S", "MS"] else args.dec_in  # output size


args.d_model = 512  # dimension of model; also the dimension of the token embeddings
args.n_heads = 512  # num of attention heads
args.e_layers = 12  # num of encoder layers
args.d_ff = 2048  # dimension of fcn in model
args.dropout = 0.05  # dropout
args.embed = None  # time features encoding, options:[timeF, fixed, learned, None]
args.activation = "gelu"  # activation

args.attn = "full"  # attention used in encoder, options:[prob, full]
args.factor = 5  # probsparse attn factor; doesn't matter unless args.attn==prob
args.distil = False  # whether to use distilling in encoder
args.output_attention = False  # whether to output attention in encoder
args.mix = False  # whether to use mixed attention


args.batch_size = 128
args.learning_rate = 0.00001
args.loss = "stock_tanh"  # What loss function to use, options:["mse", "mae", "stock_dir", "stock_dir-ns", "stock_tanh"]
args.lradj = None  # What learning rate scheduler to use: ["type3", None, "type1"]

args.optim = "AdamW"  # Adam, AdamW
args.train_epochs = 800
args.patience = 100  # For early stopping


args.scale = True  # whether to scale to mean 0, var 1
args.no_scale_mean = True  # whether to disable the mean scaling
args.inverse_output = (
    False  # whether to invert-scale the model's output before loss is calculated
)
args.inverse_pred = True  # whether to invert-scale the data label

# This is for debugging to overfit
# When True, patience doesn't matter at all and the model-state that is saved is the one after the last epoch
# When False, the model-state that is saved is the one with the highest validation-loss and we can early stop with patience
args.no_early_stop = False


# Control data split from args, either a date string like "2000-01-30" or None (for default)
args.date_start = "2012-01-01"  # Train data starts on this date, default is to go back as far as possible
args.date_end = "2020-01-01"  # Train data starts on this date, default is to go back as far as possible
args.date_test = "2019-06-01"  # Test data is data after this date, default is to use ~20% of the data as test data

args.dont_shuffle_train = True

# args.load_model_path = "stockformer_custom_ftMS_sl16_ll4_pl1_ei12_di12_co1_iFalse_dm512_nh8_el12_dl4_df2048_atfull_fc5_ebtimeF_dtFalse_mxFalse_pretrain_full_1h_0/checkpoint-pretrain.pth"


# TODO: Figure out what this is for
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print("Args in experiment:")
print(args)
setting = setting_from_args(args)
print("Setting:", setting)


strategy = "dp"  # ["ddp", "ddp_spawn", "ddp_notebook", "ddp_fork", None]
num_workers = 2  # os.cpu_count() * (strategy != "ddp_spawn")


# pl.seed_everything(seed=123, workers=True)

# Create Data Module
data_module = CustomDataModule(args, num_workers)


# Instantiate Lightning Model
exp = ExpTimeseries(args)


# Define Callbacks
callbacks = []


# Early Stop
if not args.no_early_stop:
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.000001,
            patience=args.patience,
            verbose=True,
            mode="min",
        )
    )


# Checkpoint model with lowest val lost into checkpoint.ckpt
# Additionally, checkpoint final model into last.ckpt if args.no_early_stop
callbacks.append(
    ModelCheckpoint(
        filename="checkpoint",
        save_top_k=1,
        save_last=args.no_early_stop,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
)

# Print model details
callbacks.append(ModelSummary(max_depth=2))

# Write data on predict
callbacks.append(PredTrueDateWriter("result", "epoch"))

# Stochastic Weight Averaging to improve generalization
callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2, device=None))

callbacks.append(LearningRateMonitor("epoch"))

# Print all callbacks
print(
    "Callbacks:",
    list(map(lambda x: str(type(x))[str(type(x)).rfind(".") + 1 : -2], callbacks)),
)

# Logger
logger = TensorBoardLogger(
    "lightning_logs", name=setting, flush_secs=15  # , default_hp_metric=False,
)

# Define Trainer Params
trainer_params = {
    # "auto_scale_batch_size": "power",
    # "auto_lr_find": True,
    # "fast_dev_run": True,  # For debugging
    # "profiler": "simple",  # For looking for bottlenecks
    # "detect_anomaly": True,
    # "overfit_batches": 1,
    "track_grad_norm": 2,
    "max_epochs": args.train_epochs,
    "accelerator": "gpu",
    "devices": [1],
    "auto_select_gpus": True,
    "strategy": strategy,  # Multi GPU
    "default_root_dir": f"lightning_logs/{setting}",
    "enable_model_summary": False,
    "callbacks": callbacks,
    "logger": logger,
    "log_every_n_steps": 25,
}
trainer = pl.Trainer(**trainer_params)
trainer.logger.log_hyperparams(args)

# Tune model (noop unless auto_scale_batch_size or auto_lr_find)
# tuner_result = trainer.tune(exp, datamodule=data_module)
# if "lr_find" in tuner_result:
#     tuner_result["lr_find"].plot(suggest=True)
# if "scale_batch_size" in tuner_result:
#     print("scale_batch_size:", tuner_result["scale_batch_size"])


# Train Model
trainer.fit(exp, data_module)

if not args.no_early_stop:
    exp = exp.load_from_checkpoint(
        os.path.join(trainer.log_dir, "checkpoints/checkpoint.ckpt"), config=args
    )


# Test Model
t = trainer.test(exp, data_module)

# Predict and Save Results
results = trainer.predict(exp, data_module)

print("DONE!!!! Logged in:", trainer.log_dir)
