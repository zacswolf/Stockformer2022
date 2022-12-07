import torch
from utils.tools import dotdict


def stock_loss(
    threshold: float = 0.0002,
    stock_loss_mode: str | list[str] = "lppws",
):
    # TODO: Make stock_loss_mode as list logic
    def stock_loss_closure(pred: torch.FloatTensor, true: torch.FloatTensor):
        pred_c_log, true_c_log = apply_threshold(pred, true, threshold)

        if stock_loss_mode == "lpp":
            # Log percent profit without shorting
            loss = lpp_direction(pred_c_log, true_c_log)
        elif stock_loss_mode == "lppws":
            # Log percent profit with shorting
            loss = lpp_direction_short(pred_c_log, true_c_log)
        elif "tanh":
            loss = lpp_tanh_short(pred_c_log, true_c_log)

        return loss

    return stock_loss_closure


def apply_threshold(output, other, threshold=0.0002):
    output_tresh = output[torch.abs(output) >= threshold]
    other = other[torch.abs(output) >= threshold]
    return output_tresh, other


def lpp_direction(output, log_pct_change):
    """
    Log percent profit without shorting
    """
    return -((log_pct_change * torch.sign(output))[output > 0].sum())


def lpp_direction_short(output, log_pct_change):
    """
    Log percent profit with shorting
    """
    return -((log_pct_change * torch.sign(output)).sum())


def lpp_tanh(output, log_pct_change):
    """
    Log percent profit with partial purchase
    """
    # TODO: Figure out why we are needing the constant & what to use
    return -((output * (1000 * log_pct_change).tanh())[output > 0].sum())

def lpp_tanh_short(output, log_pct_change):
    """
    Log percent profit with shorting and partial purchase
    """
    # TODO: Figure out why we are needing the constant & what to use
    return -((output * (1000 * log_pct_change).tanh()).sum())

def pct_dir(output, log_pct_change):
    return torch.sum(torch.sign(output) == torch.sign(log_pct_change)) / len(
            log_pct_change
        )
