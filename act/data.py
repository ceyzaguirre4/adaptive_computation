import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import randint
from act.utils import pad_batch


class ParityDataset(Dataset):
    def __init__(self, size, len):
        self.size = size
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.randint(0, 2, (self.size, )) * 2 - 1
        zeros = torch.randint(1, self.size, (1, ), dtype=torch.uint8)
        x[zeros[0]:] = 0.0
        # shuffle x
        idx = torch.randperm(self.size)
        x = x[[idx]].float()
        # y = torch.sum(torch.where(x == 1, x, torch.zeros(1)), 0) % 2
        y = torch.sum(((x*2)+1)//2) % 2
        return x, y.unsqueeze(0)


class ParityDataManager:
    @classmethod
    def create_dataloader(cls, cfg, train=True):
        samples_batch = int(cfg.DATALOADER.TRAIN_SAMPLES)
        if train:
            parity_dataset = ParityDataset(cfg.INPUT.DIM, samples_batch)
            return DataLoader(parity_dataset,
                              batch_size=cfg.DATALOADER.BATCH_SIZE,
                              num_workers=cfg.DATALOADER.NUM_WORKERS,
                              drop_last=True, pin_memory=True)
        else:
            parity_dataset = ParityDataset(cfg.INPUT.DIM,
                                           int(samples_batch * cfg.DATALOADER.TRAIN_TEST_PROPORTION))
            return DataLoader(parity_dataset,
                              batch_size=cfg.DATALOADER.BATCH_SIZE,
                              num_workers=cfg.DATALOADER.NUM_WORKERS,
                              drop_last=True, pin_memory=True)


class AdditionDataset(Dataset):
    def __init__(self, cfg, samples):
        self.cfg = cfg
        self.len = samples
        assert not self.cfg.INPUT.DIM % 10

    def gen_input(self):
        # TODO: maybe? avoid starting with zeros

        # difficulty = number of digits of number
        max_D = self.cfg.INPUT.DIM // 10
        D = torch.randint(1, (max_D) + 1, (1,))

        # decoded input
        dec_digits = torch.randint(0, 10, (max_D,))
        if D < max_D:
            dec_digits[D:] = 0
        dec_in = 0
        for n, i_digit in enumerate(reversed(range(0, D))):
            dec_in += (dec_digits[i_digit] * 10**n)

        enc_in = torch.zeros(self.cfg.INPUT.DIM)
        idx = dec_digits + (torch.arange(0, max_D) * 10)
        # one hot representation of encoded input
        enc_in[idx] = 1.0

        # encoded input (masked with D)
        enc_in = enc_in.view(max_D, 10)
        if D < max_D:
            enc_in[D:] = 0
        enc_in = enc_in.view(self.cfg.INPUT.DIM)

        return dec_in, enc_in

    def encode_out(self, decoded_outs):
        max_D = self.cfg.INPUT.DIM // 10

        seq_len = decoded_outs.shape[0]
        encoded = torch.zeros(seq_len, max_D+1, dtype=torch.long)
        for i_num in range(seq_len):
            num = decoded_outs[i_num]
            for i_digit, digit in enumerate(map(int, str(num.item()))):
                encoded[i_num, i_digit] = digit
            for i_digit in range(len(str(num.item())), max_D+1):
                encoded[i_num, i_digit] = 10

        return encoded.view(seq_len, -1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # how many numbers are added TODO: currently always adding at least two numbers
        seq_len = randint(2,5)

        # generating inputs
        dec_ins, enc_ins = zip(*[self.gen_input() for _ in range(seq_len)])
        dec_ins = torch.stack(dec_ins, dim=0)
        enc_ins = torch.stack(enc_ins, dim=0)

        # add to get cumsum and encode
        dec_out = dec_ins.cumsum(dim=0)
        enc_out = self.encode_out(dec_out)

        return enc_ins, enc_out, dec_out


class AdditionDataManager:
    @classmethod
    def create_dataloader(cls, cfg, train=True):
        samples_batch = int(cfg.DATALOADER.TRAIN_SAMPLES)
        if train:
            add_dataset = AdditionDataset(cfg, samples_batch)
            return DataLoader(add_dataset,
                              batch_size=cfg.DATALOADER.BATCH_SIZE,
                              num_workers=cfg.DATALOADER.NUM_WORKERS,
                              collate_fn=pad_batch,
                              drop_last=True, pin_memory=True)
        else:
            add_dataset = AdditionDataset(cfg, int(samples_batch * cfg.DATALOADER.TRAIN_TEST_PROPORTION))
            return DataLoader(add_dataset,
                              batch_size=cfg.DATALOADER.BATCH_SIZE,
                              num_workers=cfg.DATALOADER.NUM_WORKERS,
                              collate_fn=pad_batch,
                              drop_last=True, pin_memory=True)


################

def resolve_data_manager(cfg):
    if cfg.MODEL.TASK == 'parity':
        return ParityDataManager()
    elif cfg.MODEL.TASK == 'addition':
      return AdditionDataManager()
    else:
        raise NotImplementedError