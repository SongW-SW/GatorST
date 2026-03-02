import os
import random
import time
import torch
import numpy as np


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def format_time(seconds):
    if seconds <= 60:
        return '%.1fs' % seconds
    elif seconds <= 3600:
        return '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        return '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def pearson_corr(imputed_data, original_data):
    Y = original_data.reshape(-1)
    fake_Y = imputed_data.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
        np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2))
    )
    return corr


def DropData(batch_x, d_rate):
    zero_idx = (batch_x != 0).float()
    batch_x_nonzero = torch.where(batch_x == 0,
                                   torch.full_like(batch_x, -999), batch_x)
    sample_mask = torch.rand(batch_x_nonzero.shape, device=batch_x.device) <= d_rate
    batch_x_drop = torch.where(sample_mask, torch.zeros_like(batch_x_nonzero), batch_x_nonzero)

    final_mask = torch.where(batch_x_drop == 0,
                              torch.ones_like(batch_x_drop),
                              torch.zeros_like(batch_x_drop) * zero_idx)
    final_x = torch.where(batch_x_drop == -999, torch.zeros_like(batch_x), batch_x_drop)
    return final_mask, final_x
