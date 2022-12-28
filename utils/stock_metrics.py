import torch
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

    if "tanh" in stock_algo:
        if stock_algo == "tanh":
            assert target_type == "pctchange"
            return PctProfitTanh

        assert target_type == "logpctchange"

        if stock_algo == "tanhv1":
            return LogPctProfitTanhV1
        elif stock_algo == "tanhv2":
            return LogPctProfitTanhV2
        elif stock_algo == "tanhv3":
            return LogPctProfitTanhV3

        raise Exception("Invalid tanh loss")
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


class LogPctProfitTanhV1(StockAlgo):
    """
    Percent profit with investing tanh partial purchase

    V1: just uses a tanh based multiplier. This is inaccurate as the bounds for shorting is not -1 but `log(1-pctchange)/log(1+pctchange)`.
    However this quantity is near -1.
    """

    @staticmethod
    def loss(output, log_pct_change, short_filter: None | str = None):
        tanh = torch.tanh(output)
        raw = log_pct_change * tanh
        return apply_short_filter(output, raw, short_filter)

    @staticmethod
    def metric(output, log_pct_change, short_filter: None | str = None):
        # TODO: Potentially clip the negative tanh outputs: max(tanh, log(1-pctchange)/log(1+pctchange))
        tanh = np.tanh(output)
        raw = log_pct_change * tanh
        return np.exp(apply_short_filter(output, raw, short_filter).sum())

    @staticmethod
    def accumulate(output, log_pct_change, short_filter: None | str = None):
        # TODO: Potentially clip the negative tanh outputs: max(tanh, log(1-pctchange)/log(1+pctchange))
        tanh = np.tanh(output)
        raw = log_pct_change * tanh
        return np.exp(np.cumsum(apply_short_filter(output, raw, short_filter)))


class LogPctProfitTanhV2(StockAlgo):
    """
    Percent profit with investing tanh partial purchase

    V2: Scales the just negative tanh output so that they are between `log(1-pctchange)/log(1+pctchange)` and  `0`.
    """

    @staticmethod
    def loss(output, log_pct_change, short_filter: None | str = None):
        # The partial purchase multiplier is from [log(1-pctchange)/log(1+pctchange), 1]
        # mult_min is multi_min log(1-pctchange)/log(1+pctchange) and is negative
        # TODO: Look into logaddexp
        pct_change_mult = torch.exp(log_pct_change)
        mult_min = torch.log(-pct_change_mult + 2) / log_pct_change
        mult_min[mult_min != mult_min] = -1  # get rid of nan from 0/0

        tanh = torch.tanh(output)

        # Scale only the negative side of the tanh
        mult_min[tanh >= 0] = -1.0
        scaled_tanh = tanh * (-mult_min)

        raw = log_pct_change * scaled_tanh
        return apply_short_filter(output, raw, short_filter)

    @staticmethod
    def metric(output, log_pct_change, short_filter: None | str = None):
        pct_change_mult = np.exp(log_pct_change)
        mult_min = np.log(-pct_change_mult + 2) / log_pct_change
        mult_min[mult_min != mult_min] = -1

        tanh = np.tanh(output)

        mult_min[tanh >= 0] = -1.0
        scaled_tanh = tanh * (-mult_min)

        raw = log_pct_change * scaled_tanh
        return np.exp(apply_short_filter(output, raw, short_filter).sum())

    @staticmethod
    def accumulate(output, log_pct_change, short_filter: None | str = None):
        pct_change_mult = np.exp(log_pct_change)
        mult_min = np.log(-pct_change_mult + 2) / log_pct_change
        mult_min[mult_min != mult_min] = -1

        tanh = np.tanh(output)

        mult_min[tanh >= 0] = -1.0
        scaled_tanh = tanh * (-mult_min)

        raw = log_pct_change * scaled_tanh
        return np.exp(np.cumsum(apply_short_filter(output, raw, short_filter)))


class LogPctProfitTanhV3(StockAlgo):
    """
    Percent profit with investing tanh partial purchase

    V3: Scales the whole tanh output so that they are between `log(1-pctchange)/log(1+pctchange)` and  `1`.
    Note: We are most likely going to remove this version because due to the shift `output>0` will not necessarily mean buy, same with `output<0` for short
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

        # Could just use a sigmoid with 2*output
        at_sigmoid_bounds = (tanh + 1) / 2.0
        raw = log_pct_change * ((1 - mult_min) * at_sigmoid_bounds + mult_min)

        return apply_short_filter(output, raw, short_filter)

    @staticmethod
    def metric(output, log_pct_change, short_filter: None | str = None):
        pct_change_mult = np.exp(log_pct_change)
        mult_min = np.log(-pct_change_mult + 2) / log_pct_change
        mult_min[mult_min != mult_min] = -1

        tanh = np.tanh(output)

        at_sigmoid_bounds = (tanh + 1) / 2.0
        raw = log_pct_change * ((1 - mult_min) * at_sigmoid_bounds + mult_min)

        return np.exp(apply_short_filter(output, raw, short_filter).sum())

    @staticmethod
    def accumulate(output, log_pct_change, short_filter: None | str = None):
        pct_change_mult = np.exp(log_pct_change)
        mult_min = np.log(-pct_change_mult + 2) / log_pct_change
        mult_min[mult_min != mult_min] = -1

        tanh = np.tanh(output)

        at_sigmoid_bounds = (tanh + 1) / 2.0
        raw = log_pct_change * ((1 - mult_min) * at_sigmoid_bounds + mult_min)

        return np.exp(np.cumsum(apply_short_filter(output, raw, short_filter)))
