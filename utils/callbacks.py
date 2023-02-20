import pickle
from typing import Any
import pandas as pd
from pytorch_lightning.callbacks import BasePredictionWriter
import os
import torch
import numpy as np


class PredTrueDateWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        folder_path = os.path.join(trainer.log_dir, "results/")
        os.makedirs(folder_path, exist_ok=True)

        data = {}
        for dataloader_idx, (data_dl, bi) in enumerate(zip(predictions, batch_indices)):
            dataloader_data = {}
            dataset = trainer.predict_dataloaders[dataloader_idx].dataset

            for key in ["pred", "true"]:
                dataloader_data[key] = torch.concat(
                    [data_dl[i][key] for i in range(len(data_dl))]
                )

            # Date
            batch_x_raw_dates, batch_y_raw_dates = dataset.index_to_dates(
                np.concatenate(bi)
            )

            # TODO: Make safety shape assert
            batch_y_raw_dates = batch_y_raw_dates[
                :, -trainer.datamodule.config.pred_len :
            ]
            dataloader_data["date"] = batch_y_raw_dates

            data[dataset.flag] = dataloader_data

            for key in dataloader_data:
                np.save(
                    os.path.join(
                        folder_path, f"{key}_{dataset.flag}_{trainer.global_rank}.npy"
                    ),
                    dataloader_data[key],
                )


# class PredTrueDateWriterV2(BasePredictionWriter):
#     def __init__(self, output_dir, write_interval="epoch"):
#         super().__init__(write_interval)
#         self.output_dir = output_dir

#     def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
#         # this will create N (num processes) files in `output_dir` each containing
#         # the predictions of it's respective rank
#         folder_path = os.path.join(trainer.log_dir, "results/")
#         os.makedirs(folder_path, exist_ok=True)

#         tpd_dict_tuple: dict[str, tuple[Any, Any, Any]] = {}

#         data = {}
#         for dataloader_idx, (data_dl, bi) in enumerate(zip(predictions, batch_indices)):
#             dataloader_data = {}
#             dataset = trainer.predict_dataloaders[dataloader_idx].dataset

#             for key in ["pred", "true"]:
#                 dataloader_data[key] = torch.concat(
#                     [data_dl[i][key] for i in range(len(data_dl))]
#                 )

#             # Date
#             batch_x_raw_dates, batch_y_raw_dates = dataset.index_to_dates(
#                 np.concatenate(bi)
#             )

#             # TODO: Make safety shape assert
#             batch_y_raw_dates = batch_y_raw_dates[
#                 :, -trainer.datamodule.config.pred_len :
#             ]
#             dataloader_data["date"] = batch_y_raw_dates

#             data[dataset.flag] = dataloader_data

#             for key in dataloader_data:
#                 np.save(
#                     os.path.join(
#                         folder_path, f"{key}_{dataset.flag}_{trainer.global_rank}.npy"
#                     ),
#                     dataloader_data[key],
#                 )


#         for data_group in ["train", "val", "test"]:

#             dp = [
#                 data[data_group]["true"][:, 0, 0],
#                 data[data_group]["pred"][:, 0, 0],
#                 data[data_group]["date"][:, 0],
#             ]
#             tpd_dict_tuple[data_group] = dp
#             s = np.argsort(tpd_dict_tuple[data_group][2], axis=None)
#             tpd_dict_tuple[data_group] = list(
#                 map(lambda x: x[s], tpd_dict_tuple[data_group])
#             )

#             tpd_dict_tuple[data_group][2] = pd.DatetimeIndex(
#                 tpd_dict_tuple[data_group][2], tz="UTC"
#             )

#             # # Override trues with df target data to get original numerical precision
#             # if not ("mse" in args.loss and not args.inverse_output) and df is not None:
#             #     print("OVERRIDING trues with df target")
#             #     df_data_group = df.loc[tpd_dict_tuple[data_group][2]]
#             #     t = args.target.split("_")
#             #     df_target = df_data_group[t[0]][t[1]].to_numpy()
#             #     tpd_dict_tuple[data_group][0] = df_target


#         tpd_dict: dict[str, dict[str, Any]] = {}
#         for data_group in tpd_dict_tuple:
#             tpd_dict[data_group] = {
#                 "trues": tpd_dict_tuple[data_group][0],
#                 "preds": tpd_dict_tuple[data_group][1],
#                 "dates": tpd_dict_tuple[data_group][2],
#             }

#         get_metrics(args: dotdict | None, pred: np.ndarray, true: np.ndarray, thresh: float = 0.0)

#         with open(os.path.join(folder_path, "tpd_dict.pickle"), "wb") as handle:
#             pickle.dump(tpd_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         return tpd_dict
