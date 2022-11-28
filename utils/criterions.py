import torch
from utils.tools import dotdict

def stock_loss(args: dotdict, stock_lock_thresh:float = .0002, stock_loss_mode:str|list[str] = "lppws"):
    # TODO: Make stock_loss_mode as list logic
    def stock_loss_closure(pred: torch.FloatTensor, true: torch.FloatTensor):
        true_c_log = true[torch.abs(pred) >= stock_lock_thresh]
        pred_c_log = pred[torch.abs(pred) >= stock_lock_thresh]

        if stock_loss_mode == "lpp":
            # Log percent profit without shorting
            loss = -((true_c_log * torch.sign(pred_c_log))[pred_c_log > 0].sum())
        elif stock_loss_mode == "lppws":
            # Log percent profit with shorting
            loss = -((true_c_log * torch.sign(pred_c_log)).sum())
        elif "tanh":
            # Log percent profit with shorting with partial purchase
            # TODO: Figure out why we are needing the constant & what to use
            loss = -((true_c_log * (1000*pred_c_log).tanh()).sum())

        return loss
    
    return stock_loss_closure


