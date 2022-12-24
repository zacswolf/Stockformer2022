import torch
from torchmetrics import Metric
import numpy as np


class StockAlgo:
    @staticmethod
    def loss(output, pct_change, short_filter: None | str = None):
        raise NotImplementedError

    @staticmethod
    def metric(output, pct_change, short_filter: None | str = None):
        raise NotImplementedError

    @staticmethod
    def accumulate(output, pct_change, short_filter: None | str = None):
        raise NotImplementedError


def get_stock_algo(target_type, stock_loss_mode) -> StockAlgo:
    if target_type != "logpctchange" and target_type != "pctchange":
        raise Exception(f"Invalid Target Type: {target_type}")

    stock_algo = stock_loss_mode.split("-")[0]

    if "tanh" == stock_algo:
        return LogPctProfitTanh if target_type == "logpctchange" else PctProfitTanh
    elif "dir" == stock_algo:
        return (
            LogPctProfitDirection
            if target_type == "logpctchange"
            else PctProfitDirection
        )

    raise Exception(f"Invalid Stock Loss Mode: {stock_loss_mode}")


def get_short_filter(stock_loss_mode: str) -> None | str:
    stock_algo_split = stock_loss_mode.split("-")
    if len(stock_algo_split) == 1:
        return None
    else:
        assert stock_algo_split[-1] in ["ns", "os"]
        return stock_algo_split[-1]


def apply_short_filter(output, raw, short_filter: None | str):
    if short_filter is None:
        return raw
    elif short_filter == "ns":
        return raw[output > 0]
    elif short_filter == "os":
        return raw[output < 0]
    raise Exception(f"Invalid short filter: {short_filter}")


# def apply_threshold_loss(output, other, threshold=0.0002):
#     output_tresh = output[torch.abs(output) >= threshold]
#     other = other[torch.abs(output) >= threshold]
#     return output_tresh, other


def apply_threshold_metric(output, other, threshold=0.0002):
    output_tresh = output[np.abs(output) >= threshold]
    other = other[np.abs(output) >= threshold]
    return output_tresh, other


def pct_direction(output, pct_change):
    return np.sum(np.sign(output) == np.sign(pct_change)) / len(pct_change)


class PctProfitDirection(StockAlgo):
    """
    Percent profit with investing everything strategy.
    """

    @staticmethod
    def loss(output, pct_change, short_filter: None | str = None):
        raw = pct_change * torch.sign(output) + 1
        return apply_short_filter(output, raw, short_filter)

    @staticmethod
    def metric(output, pct_change, short_filter: None | str = None):
        raw = pct_change * np.sign(output) + 1
        return apply_short_filter(output, raw, short_filter).prod()

    @staticmethod
    def accumulate(output, pct_change, short_filter: None | str = None):
        raw = pct_change * np.sign(output) + 1
        return np.cumprod(apply_short_filter(output, raw, short_filter))


class LogPctProfitDirection(StockAlgo):
    """
    Percent profit with investing everything strategy.
    """

    @staticmethod
    def loss(output, log_pct_change, short_filter: None | str = None):
        raw = log_pct_change * torch.sign(output)
        return apply_short_filter(output, raw, short_filter)

    @staticmethod
    def metric(output, log_pct_change, short_filter: None | str = None):
        raw = log_pct_change * np.sign(output)
        return np.exp(apply_short_filter(output, raw, short_filter).sum())

    @staticmethod
    def accumulate(output, log_pct_change, short_filter: None | str = None):
        raw = log_pct_change * np.sum(output)
        return np.exp(np.cumsum(apply_short_filter(output, raw, short_filter)))


class PctProfitTanh(StockAlgo):
    """
    Percent profit with investing tanh partial purchase
    """

    @staticmethod
    def loss(output, pct_change, short_filter: None | str = None):
        raw = (pct_change * torch.tanh(output)) + 1
        return apply_short_filter(output, raw, short_filter)

    @staticmethod
    def metric(output, pct_change, short_filter: None | str = None):
        raw = pct_change * np.tanh(output) + 1
        return apply_short_filter(output, raw, short_filter).prod()

    @staticmethod
    def accumulate(output, pct_change, short_filter: None | str = None):
        raw = pct_change * np.tanh(output) + 1
        return np.cumprod(apply_short_filter(output, raw, short_filter))


class LogPctProfitTanh(StockAlgo):
    """
    Percent profit with investing tanh partial purchase
    """

    @staticmethod
    def loss(output, log_pct_change, short_filter: None | str = None):
        # The partial purchase multiplier is from [log(1-pctchange)/log(1+pctchange), 1]
        # mult_min is multi_min log(1-pctchange)/log(1+pctchange) and is negative
        # Look into logaddexp
        pct_change_mult = torch.exp(log_pct_change)
        mult_min = torch.log(-pct_change_mult + 2) / log_pct_change
        mult_min[mult_min != mult_min] = -1  # get rid of nan from 0/0

        tanh = torch.tanh(output)

        # METHOD 1
        # # Could just use a sigmoid
        # at_sigmoid_bounds = (torch.tanh(output) + 1) / 2.0
        # raw = log_pct_change * ((1 - mult_min) * at_sigmoid_bounds + mult_min)

        # METHOD 2
        # mult_min = -mult_min
        # mult_min[tanh >= 0] = 1.0
        # tanh = tanh * mult_min

        # METHOD 3: JUST LEAVE THE TANH


        raw = log_pct_change * tanh
        return apply_short_filter(output, raw, short_filter)

    @staticmethod
    def metric(output, log_pct_change, short_filter: None | str = None):
        pct_change_mult = np.exp(log_pct_change)
        mult_min = np.log(-pct_change_mult + 2) / log_pct_change
        mult_min[mult_min != mult_min] = -1
        at_sigmoid_bounds = (np.tanh(output) + 1) / 2.0
        raw = log_pct_change * ((1 - mult_min) * at_sigmoid_bounds + mult_min)
        return np.exp(apply_short_filter(output, raw, short_filter).sum())

    @staticmethod
    def accumulate(output, log_pct_change, short_filter: None | str = None):
        pct_change_mult = np.exp(log_pct_change)
        mult_min = np.log(-pct_change_mult + 2) / log_pct_change
        mult_min[mult_min != mult_min] = -1
        at_sigmoid_bounds = (np.tanh(output) + 1) / 2.0
        raw = log_pct_change * ((1 - mult_min) * at_sigmoid_bounds + mult_min)
        return np.exp(np.cumsum(apply_short_filter(output, raw, short_filter)))


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
