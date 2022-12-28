# We are not using anything in this file at the moment
import torch
from torchmetrics import Metric

from utils.stock_metrics import get_short_filter, get_stock_algo


def get_stock_loss(target_type, stock_loss_mode, threshold=0.0) -> Metric:
    if target_type == "pctchange":
        return PctProfit(target_type, stock_loss_mode, threshold)
    elif target_type == "logpctchange":
        return LogPctProfit(target_type, stock_loss_mode, threshold)


class PctProfit(Metric):
    @property
    def is_differentiable(self) -> bool:
        return True

    def __init__(self, target_type, stock_loss_mode, threshold=0.0):
        super().__init__()
        assert target_type == "pctchange"

        self.add_state(
            "pct_profit", default=torch.tensor(1, dtype=float), dist_reduce_fx="mean"
        )

        self.loss_fnt = get_stock_algo(target_type, stock_loss_mode)
        self.short_filter = get_short_filter(stock_loss_mode)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.pct_profit *= self.loss_fnt.loss(
            preds, target, short_filter=self.short_filter
        ).prod()

    def compute(self):
        return -self.pct_profit


class LogPctProfit(Metric):
    @property
    def is_differentiable(self) -> bool:
        return True

    def __init__(self, target_type, stock_loss_mode, threshold=0.0):
        super().__init__()
        assert target_type == "logpctchange"

        self.add_state(
            "log_pct_profit", default=torch.tensor(0, dtype=float), dist_reduce_fx="sum"
        )
        # self.add_state("n_observations", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold

        self.loss_fnt = get_stock_algo(target_type, stock_loss_mode)
        self.short_filter = get_short_filter(stock_loss_mode)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.log_pct_profit += self.loss_fnt.loss(
            preds, target, short_filter=self.short_filter
        ).sum()

        # self.n_observations += preds.numel()

    def compute(self):
        return -self.log_pct_profit

    def get_stock_metric(self):
        return self.loss_fnt
