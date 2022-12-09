import torch
import numpy as np


def stock_loss(
    threshold: float = 0.0002,
    stock_loss_mode: str = "lpp",
):
    def stock_loss_closure(pred: torch.FloatTensor, true: torch.FloatTensor):
        pred_c_log, true_c_log = apply_threshold_loss(pred, true, threshold)

        if stock_loss_mode == "lpp":
            # Log percent profit with shorting
            loss = PctProfitDirection.loss(pred_c_log, true_c_log, short_filter=0)
        elif stock_loss_mode == "lppns":
            # Log percent profit without shorting
            loss = PctProfitDirection.loss(pred_c_log, true_c_log, short_filter=1)
        elif "tanh":
            loss = PctProfitDirection.loss(pred_c_log, true_c_log, short_filter=0)

        raise Exception(f"Invalid Loss: {stock_loss_mode}")

    return stock_loss_closure


def apply_threshold_loss(output, other, threshold=0.0002):
    output_tresh = output[torch.abs(output) >= threshold]
    other = other[torch.abs(output) >= threshold]
    return output_tresh, other


def apply_threshold_metric(output, other, threshold=0.0002):
    output_tresh = output[np.abs(output) >= threshold]
    other = other[np.abs(output) >= threshold]
    return output_tresh, other


class StockLogits:
    @staticmethod
    def loss(output, log_pct_change, short_filter: int = 0):
        raise NotImplementedError

    @staticmethod
    def metric(output, log_pct_change, short_filter: int = 0):
        raise NotImplementedError

    @staticmethod
    def accumulate(output, log_pct_change, short_filter: int = 0):
        raise NotImplementedError


class PctProfitDirection(StockLogits):
    """
    Percent profit with investing everything strategy.
    """

    @staticmethod
    def loss(output, log_pct_change, short_filter: int = 0):
        raw = log_pct_change * torch.sign(output)
        if short_filter == 1:
            return -raw[output > 0].sum()
        elif short_filter == 2:
            return -raw[output < 0].sum()
        return -raw.sum()

    @staticmethod
    def metric(output, log_pct_change, short_filter: int = 0):
        raw = log_pct_change * np.sign(output)
        if short_filter == 1:
            return np.exp(raw[output > 0].sum())
        elif short_filter == 2:
            return np.exp(raw[output < 0].sum())
        return np.exp(raw.sum())

    @staticmethod
    def accumulate(output, log_pct_change, short_filter: int = 0):
        raw = log_pct_change * np.sign(output)
        if short_filter == 1:
            return np.exp(np.cumsum(raw[output > 0]))
        elif short_filter == 2:
            return np.exp(np.cumsum(raw[output < 0]))
        return np.exp(np.cumsum(raw))


class PctProfitTanh(StockLogits):
    """
    Percent profit with investing tanh partial purchase
    """

    @staticmethod
    def loss(output, log_pct_change, short_filter: int = 0):
        raw = output * (1000 * log_pct_change).tanh()
        if short_filter == 1:
            return -raw[output > 0].sum()
        elif short_filter == 2:
            return -raw[output < 0].sum()
        return -raw.sum()

    @staticmethod
    def metric(output, log_pct_change, short_filter: int = 0):
        raw = output * np.tanh(1000 * log_pct_change)
        if short_filter == 1:
            return np.exp(raw[output > 0].sum())
        elif short_filter == 2:
            return np.exp(raw[output < 0].sum())
        return np.exp(raw.sum())

    @staticmethod
    def accumulate(output, log_pct_change, short_filter: int = 0):
        raw = output * np.tanh(1000 * log_pct_change)
        if short_filter == 1:
            return np.exp(np.cumsum(raw[output > 0]))
        elif short_filter == 2:
            return np.exp(np.cumsum(raw[output < 0]))
        return np.exp(np.cumsum(raw))


class PctDirection(StockLogits):
    """
    Percentage of output with the correct sign
    """

    @staticmethod
    def metric(output, log_pct_change):
        return np.sum(np.sign(output) == np.sign(log_pct_change)) / len(log_pct_change)
