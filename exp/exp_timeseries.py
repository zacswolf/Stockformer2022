import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.Basic import MLP, NLinear
from models.Informer import Informer, InformerStack
from models.Stockformer import Stockformer
from utils.criterions import stock_loss


class ExpTimeseries(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # pl makes self.learning_rate special
        self.learning_rate = config.learning_rate

        self.criterion = self._select_criterion()
        self._build_model()
        # self.save_hyperparameters()

    def _build_model(self):
        model_dict = {
            "informer": Informer,
            "informerstack": InformerStack,
            "mlp": MLP,
            "stockformer": Stockformer,
            "nlinear": NLinear,
        }
        assert (
            self.config.model in model_dict
        ), f"Invalid config.model: {self.config.model}, options: {list(model_dict.keys())}"
        self.model = model_dict[self.config.model](self.config).float()

        # Load model
        if self.config.load_model_path is not None:
            self.load_from_checkpoint(self.config.load_model_path)

    def _select_criterion(self):
        if "stock" in self.config.loss:
            _, stock_loss_mode = self.config.loss.split("_")
            assert (
                self.config.target.split("_")[1] == "pctchange"
            ), "Can't use stock loss unless target is pctchange"
            assert not (
                self.config.scale and not self.config.inverse
            ), "Can't use stock loss when config.scale==True and config.inverse==False."
            criterion = stock_loss(self.config, stock_loss_mode=stock_loss_mode)
            self.reduce_fx = "sum"
        else:
            assert self.config.loss == "mse"
            criterion = nn.MSELoss()
            self.reduce_fx = "mean"
        return criterion

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch

        pred, true, _ = self._process_one_batch(
            self.trainer.datamodule.data_train,
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            ds_index=None,
        )
        loss = self.criterion(pred, true)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            reduce_fx=self.reduce_fx,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # validation_step defines the validation loop. It is independent of forward
        batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch

        pred, true, _ = self._process_one_batch(
            self.trainer.datamodule.data_val,
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            ds_index=None,
        )
        loss = self.criterion(pred, true)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            reduce_fx=self.reduce_fx,
        )

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # test_step defines the test loop. It is independent of forward
        batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch

        data_sets = [
            self.trainer.datamodule.data_test,
            self.trainer.datamodule.data_train,
            self.trainer.datamodule.data_val,
        ]

        pred, true, _ = self._process_one_batch(
            data_sets[dataloader_idx],
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            ds_index=None,
        )
        loss = self.criterion(pred, true)

        # if dataloader_idx == 0:
        self.log(
            "test_loss",
            loss,
            reduce_fx=self.reduce_fx,
            sync_dist=False,
        )
        # elif dataloader_idx == 1:
        #     self.log("test_loss_trn", loss, sync_dist=False)
        # elif dataloader_idx == 2:
        #     self.log("test_loss_val", loss, sync_dist=False)

        # self.log(
        #     "test_loss",
        #     {f"ds_{0}": loss, f"ds_{1}": loss},
        #     reduce_fx=self.reduce_fx,
        #     sync_dist=False,
        #     add_dataloader_idx=False,
        # )

        # return {
        #     "pred": pred,
        #     "true": true,
        #     # "batch_idx": batch_idx,
        #     # "dataloader_idx": dataloader_idx,
        # }

    # def test_epoch_end(self, outputs):
    #     return outputs

    # def on_test_epoch_end(self, outputs=None):
    #     pass

    # def on_test_end(self):
    #     pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch

        data_sets = [
            self.trainer.datamodule.data_train,
            self.trainer.datamodule.data_val,
            self.trainer.datamodule.data_test,
        ]

        pred, true, _ = self._process_one_batch(
            data_sets[dataloader_idx],
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            ds_index=None,
        )

        return {
            "pred": pred,
            "true": true,
        }

        # return pred, true, batch_idx, dataloader_idx  # , dates.tolist()

    # def on_predict_epoch_end(self, outputs):
    #     data = []
    #     for dataloader_idx, data_dl in enumerate(outputs):
    #         dataloader_data = {}
    #         for key in data_dl[0]:
    #             dataloader_data[key] = torch.concat(
    #                 [data_dl[i][key] for i in range(len(data_dl))]
    #             )
    #         # print(dataloader_data.shape)
    #         data.append(dataloader_data)
    #         # for batch_idx, data_bi in enumerate(data_dl):
    #     return data

    # def on_predict_end(self):
    #     pass
    #     pass

    def _process_one_batch(
        self,
        dataset_object,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        ds_index=None,
    ):
        # Decoder input if self.config.dec_in
        dec_inp = None
        # if self.config.dec_in and (
        #     self.config.padding == 0 or self.config.padding == 1
        # ):
        #     # FF: dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
        #     dec_inp = torch.full(
        #         [batch_y.shape[0], self.config.pred_len, batch_y.shape[-1]],
        #         self.config.padding,
        #     ).float()
        #     dec_inp = (
        #         torch.cat([batch_y[:, : self.config.label_len, :], dec_inp], dim=1)
        #         .float()
        #         .to(self.device)
        #     )

        # Encoder - Decoder
        if self.config.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.config.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.config.features == "MS" else 0

        # if ds_index is None:
        batch_y = batch_y[:, -self.config.pred_len :, f_dim:]
        return outputs, batch_y, None
        # else:
        #     batch_x_raw_dates, batch_y_raw_dates = dataset_object.index_to_dates(
        #         ds_index
        #     )
        #     assert batch_y_raw_dates.shape == batch_y.shape[0:2]
        #     batch_y = batch_y[:, -self.config.pred_len :, f_dim:].to(self.device)
        #     batch_y_raw_dates = batch_y_raw_dates[:, -self.config.pred_len :]
        #     return outputs, batch_y, batch_y_raw_dates

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
