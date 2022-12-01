from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.Informer import Informer, InformerStack
from models.Basic import NLinear, MLP
from models.Stockformer import Stockformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.criterions import stock_loss

import numpy as np

import torch
import torch.nn as nn
from torch import optim

import os
import time
import json

import warnings

warnings.filterwarnings("ignore")


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            "informer": Informer,
            "informerstack": InformerStack,
            "mlp": MLP,
            "stockformer": Stockformer,
            "nlinear": NLinear,
        }

        # Use stack layers for encoder layers if using informerstack
        self.args.e_layers = (
            self.args.s_layers
            if self.args.model == "informerstack"
            else self.args.e_layers
        )

        assert (
            self.args.model in model_dict
        ), f"Invalid args.model: {self.args.model}, options: {list(model_dict.keys())}"
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if "stock" in self.args.loss:
            _, stock_loss_mode = self.args.loss.split("_")
            assert (
                self.args.target.split("_")[1] == "pctchange"
            ), "Can't use stock loss unless target is pctchange"
            assert not (
                self.args.scale and not self.args.inverse
            ), "Can't use stock loss when args.scale==True and args.inverse==False."
            criterion = stock_loss(self.args, stock_loss_mode=stock_loss_mode)
        else:
            assert self.args.loss == "mse"
            criterion = nn.MSELoss()
        return criterion

    def _select_scheduler(self, optimizer):
        if self.args.lradj == "type1":
            lmbda = lambda epoch: 0.5
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer, lr_lambda=lmbda, verbose=True
            )
        elif self.args.lradj == "type3":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                patience=2,
                threshold=1e-2,
                cooldown=0,
                verbose=True,
            )
        else:
            scheduler = None
        return scheduler

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(
            vali_loader
        ):
            pred, true, _ = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, ds_index=None
            )
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Save args
        with open(os.path.join(path, "args.json"), "w") as convert_file:
            convert_file.write(json.dumps(self.args))

        time_now = time.time()

        train_steps = len(train_loader)

        early_stopping = None
        if not self.args.no_early_stop:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            if epoch == 0:
                for param_group in model_optim.param_groups:
                    param_group["lr"] = 1e-8
            elif epoch == 1:
                for param_group in model_optim.param_groups:
                    param_group["lr"] = self.args.learning_rate
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(
                train_loader
            ):
                iter_count += 1

                model_optim.zero_grad()
                pred, true, _ = self._process_one_batch(
                    train_data,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    ds_index=None,
                )
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test"
                " Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            if not self.args.no_early_stop:
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # adjust_learning_rate(model_optim, epoch+1, self.args)
            if scheduler is not None:
                scheduler.step(metrics=vali_loss)

        if self.args.no_early_stop:
            # This is only for debugging
            print("Saving overfitted model")
            # os.rename(os.path.join(path, 'checkpoint.pth'), os.path.join(path, 'checkpoint-real.pth'))
            torch.save(self.model.state_dict(), os.path.join(path, "checkpoint.pth"))
        else:
            best_model_path = os.path.join(path, "checkpoint.pth")
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, flag="test", inverse=True):
        # Enable inverse if scale
        inverse_og = self.args.inverse
        self.args.inverse = self.args.scale and inverse

        data, loader = self._get_data(flag=flag)

        self.model.eval()

        preds = []
        trues = []
        raw_dates = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, ds_index) in enumerate(
            loader
        ):
            pred, true, rdates = self._process_one_batch(
                data, batch_x, batch_y, batch_x_mark, batch_y_mark, ds_index=ds_index
            )
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            raw_dates.append(rdates)

        assert len(preds) == len(trues)
        preds = np.array(preds)
        trues = np.array(trues)
        raw_dates = np.array(raw_dates)
        print(flag, "shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        raw_dates = raw_dates.reshape(-1, raw_dates.shape[-1])
        print(flag, "shape:", preds.shape, trues.shape)

        # Result save
        folder_path = os.path.join("./results/", setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save args
        with open(os.path.join(folder_path, "args.json"), "w") as convert_file:
            convert_file.write(json.dumps(self.args))

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"{flag} mse:{mse}, mae:{mae}")

        # Save metrics
        with open(os.path.join(folder_path, "results.txt"), "a") as f:
            f.write(f"{setting}\t{flag}\nmse:{mse}, mae:{mae}\n\n")
        np.save(
            os.path.join(folder_path, f"metrics_{flag}.npy"),
            np.array([mae, mse, rmse, mape, mspe]),
        )

        # Save pred & true & raw dates
        np.save(os.path.join(folder_path, f"pred_{flag}.npy"), preds)
        np.save(os.path.join(folder_path, f"true_{flag}.npy"), trues)
        np.save(os.path.join(folder_path, f"date_{flag}.npy"), raw_dates)
        self.args.inverse = inverse_og
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, "checkpoint.pth")
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        preds = []
        # pred_trues = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(
            pred_loader
        ):
            pred, true, _ = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, ds_index=None
            )
            preds.append(pred.detach().cpu().numpy())
            # pred_trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = os.path.join("./results/", setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, "real_prediction.npy"), preds)

        return

    def _process_one_batch(
        self,
        dataset_object,
        batch_x,
        batch_y,
        batch_x_mark,
        batch_y_mark,
        ds_index=None,
    ):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Decoder input if self.args.dec_in
        dec_inp = None
        if self.args.dec_in and (self.args.padding == 0 or self.args.padding == 1):
            # FF: dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.full(
                [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]],
                self.args.padding,
            ).float()
            dec_inp = (
                torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                .float()
                .to(self.device)
            )

        # Encoder - Decoder
        with torch.cuda.amp.autocast(enabled=self.args.use_amp):
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features == "MS" else 0

        if ds_index is None:
            batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
            return outputs, batch_y, None
        else:
            batch_x_raw_dates, batch_y_raw_dates = dataset_object.index_to_dates(
                ds_index
            )
            assert batch_y_raw_dates.shape == batch_y.shape[0:2]
            batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
            batch_y_raw_dates = batch_y_raw_dates[:, -self.args.pred_len :]
            return outputs, batch_y, batch_y_raw_dates
