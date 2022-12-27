import torch
import pytorch_lightning as pl

from models.Basic import MLP
from models.Lstm import LSTM
from models.Informer import Informer, InformerStack
from models.Stockformer import Stockformer
from utils.stock_metrics import get_stock_loss, get_stock_algo
from torchmetrics import MeanSquaredError, MeanAbsoluteError


class ExpTimeseries(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # pl makes self.learning_rate special
        self.learning_rate = config.learning_rate

        # Torch metrics has a state that resets but val and train can be called in unison so we split
        self.train_criterion = self._select_criterion()
        self.other_criterion = self._select_criterion()

        self._build_model()
        # self.save_hyperparameters()

    def _build_model(self):
        model_dict = {
            "informer": Informer,
            "informerstack": InformerStack,
            "mlp": MLP,
            "stockformer": Stockformer,
            "lstm": LSTM,
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
            # Using Stock Loss
            _, stock_loss_mode = self.config.loss.split("_")
            target_type = self.config.target.split("_")[1]
            assert (
                target_type == "pctchange" or target_type == "logpctchange"
            ), "Can't use stock loss unless target is pctchange or logpctchange"
            assert (
                self.config.scale
                and self.config.inverse_pred
                and not self.config.inverse_output
            ), "Can't use stock loss without scale, inverse pred, and not inverse output"

            criterion = get_stock_algo(target_type, stock_loss_mode)
            print("criterion:", criterion)
            return lambda x, y: -criterion.loss(x, y).mean()
            # return lambda x, y: -LogPctProfitTanhV1.loss(x, y).mean()
            # return get_stock_loss(target_type, stock_loss_mode, threshold=0.0)
        elif self.config.loss == "mae":
            assert (
                self.config.scale
                and self.config.inverse_pred
                and self.config.inverse_output
            ), "Can't use mae loss without scale, inverse pred, and inverse output"
            return MeanAbsoluteError()
        elif self.config.loss == "mse":
            assert (
                self.config.scale
                and self.config.inverse_pred
                and self.config.inverse_output
            ), "Can't use mse loss without scale, inverse pred, and inverse output"
            return MeanSquaredError()
        raise Exception(f"Invalid loss: {self.config.loss}")

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
        loss = self.train_criterion(pred, true)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

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

        if dataloader_idx == 0:
            # Actual val dataset
            assert self.trainer.val_dataloaders[0].dataset.flag == "val"
            loss = self.other_criterion(pred, true)
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                add_dataloader_idx=False,
            )
            return loss
        elif dataloader_idx == 1:
            # TODO: If we are using torch metrics we should create an additional loss function
            # Test dataset
            assert self.trainer.val_dataloaders[1].dataset.flag == "test"
            loss = self.other_criterion(pred, true)
            self.log(
                "test_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                add_dataloader_idx=False,
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
        loss = self.other_criterion(pred, true)

        # if dataloader_idx == 0:
        self.log(
            "test_loss",
            loss,
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

        # dataset = self.trainer.predict_dataloaders[dataloader_idx].dataset
        # batch_x_raw_date, batch_y_raw_date = dataset.index_to_dates(batch_idx)

        return {
            "pred": pred,
            "true": true,
        }

    # def on_predict_epoch_end(self, results):
    #     pass

    # def on_predict_end(self):
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
        if self.config.inverse_output:
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
        if self.config.optim == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        if self.config.lradj == "type1":
            lmbda = lambda epoch: 0.5
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer, lr_lambda=lmbda, verbose=True
            )
        elif self.config.lradj == "type3":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=2,
                threshold=0,
                cooldown=0,
                verbose=True,
                min_lr=1e-8,
            )
            scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",  # called after each training step
                "monitor": "val_loss",
            }
        else:
            return optimizer

        return [optimizer], [scheduler]

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu,
    #     using_native_amp,
    #     using_lbfgs,
    # ):
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = lr_scale * self.learning_rate

    #     optimizer.step()
    #     optimizer.zero_grad()
