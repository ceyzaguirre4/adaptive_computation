import torch
import shutil
from pathlib import Path


##############################
# pad sequences to same length
##############################
# (based on https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8)
def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=vec.dtype)], dim=dim)


def pad_batch(batch):
    # find longest sequence
    max_len = max((tup[0].shape[0] for tup in  batch))
    # pad according to max_len
    batch = [(pad_tensor(elem, pad=max_len, dim=0) for elem in tup) for tup in batch]

    # stack all
    # stacked = [torch.stack([list(tup)[n] for tup in batch], dim=0) for n in range(3)]
    stacked = [torch.stack(elems) for elems in zip(*batch)]
    y_s = stacked[1]

    mask = torch.where(y_s.sum(dim=2) >= 1, torch.ones(1), torch.zeros(1))
    mask[:,0]=0       # dont count first element in accuracy or deepmasked loss
    return (*stacked, mask)


#####################################
# get path for each saved/loaded file
#####################################
def get_path(path, prefix=""):
    base_path = Path(".") / path
    if prefix == 'log_':
        new_path = base_path.parent / (prefix + str(base_path.stem) + '.csv')
    else:
        new_path = base_path.parent / (prefix + str(base_path.name))
    return str(new_path)


###############################
# saving and loading  functions
###############################
def save_checkpoint(state, is_best, cfg):
    base_path = get_path(cfg.SAVE_PATH)
    torch.save(state, base_path)
    if is_best:
        best_path = get_path(cfg.SAVE_PATH, "best_")
        shutil.copyfile(base_path, best_path)


def load_checkpoint(load_path_str, model, optimizer, cfg, device='cpu'):
    print("=> loading checkpoint '{}'".format(load_path_str))
    if cfg.LOAD_BEST:
        load_path = get_path(load_path_str, 'best_')
        checkpoint = torch.load(load_path, map_location=device)
    else:
        load_path = get_path(load_path_str)
        checkpoint = torch.load(load_path, map_location=device)

    # overwrite options
    cfg.merge_from_list(["START_EPOCH", checkpoint['epoch'], "BEST_PRECISION", checkpoint['best_precision']])
    if not load_path_str == cfg.SAVE_PATH:
        # copy old log file to new path
        original_log_path = get_path(load_path_str, 'log_')
        new_log_path = get_path(cfg.SAVE_PATH, 'log_')
        shutil.copyfile(original_log_path, new_log_path)

        # copy old "best" file to new path
        original_best_path = get_path(load_path_str, 'best_')
        new_best_path = get_path(cfg.SAVE_PATH, 'best_')
        shutil.copyfile(original_best_path, new_best_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(load_path, checkpoint['epoch']))
